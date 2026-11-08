#include "SemanticAnalyzer.h"
#include "SemanticOptions.h"
#include <set>
#include <stdexcept>

// 语义分析整体流程：
// analyze 创建结果 Model，Analysis::run 按源码顺序访问语法树。
// 声明与语句递归向下处理；表达式的类型和常量信息由子节点向父节点返回。
// 所有语义结果保存在 Model 中，诊断则追加到词法、语法共用的 words.errors。
// 这里遍历的是含文法包装节点和终结符的语法树，访问孩子时必须区分两者。
namespace semantic {
namespace {
using Node = syntax::Node;
// 无符号关联的哨兵值，与 Expression::symbol 的默认值保持一致。
// 例如数字字面量有类型和值，但没有对应的变量声明。
constexpr std::size_t noSymbol = static_cast<std::size_t>(-1);

// 只看直接孩子，不能递归把函数体中的名字当作函数名。
const Node *rule(const Node &node, const std::string &name) {
    for (const auto &child : node.children)
        if (!child->token && child->name == name) return child.get();
    return nullptr;
}
// 按 Token 类别查找直接终结符，例如 IDENFR、STRCON。找不到返回 nullptr。
const Node *terminal(const Node &node, const std::string &kind) {
    for (const auto &child : node.children)
        if (child->token && child->token->kind == kind) return child.get();
    return nullptr;
}
// 按源码文本查找直接终结符，例如 return、[、++；text 为空时取第一个。
// rule/terminal/token 只负责查找，调用方按照 Parser 生成的节点结构解释结果。
const lexical::Token *token(const Node &node, const std::string &text = "") {
    for (const auto &child : node.children)
        if (child->token && (text.empty() || child->token->text == text)) return &*child->token;
    return nullptr;
}
// 将声明中的 BType/FuncType 转成内部类型，此处只决定基本类型。
// 数组标记由声明或形参处是否出现 [ 决定，不在这里判断。
Type declaredType(const Node &node) {
    const auto *word = token(node);
    if (!word) return {};
    if (word->text == "int") return {BaseType::Int, false};
    if (word->text == "char") return {BaseType::Char, false};
    if (word->text == "void") return {BaseType::Void, false};
    throw std::runtime_error("semantic stage supports the assignment's int/char/void grammar only");
}
// 普通类型相等要求基本类型、数组标记都相同；不比较 const/static 属性。
bool sameType(Type a, Type b) { return a.base == b.base && a.array == b.array; }

// ===== 考试定位 1：常量值的位宽 =====
// 对齐参考实现：char 是无符号 8 位，int 是有符号 32 位。
// 中间计算用 int64_t，最后截断；不能用宿主机的 char 正负号决定课程语义。
std::int64_t narrow(std::int64_t value, Type type) {
    if (type.base == BaseType::Char) return static_cast<std::uint64_t>(value) & 255U;
    const auto bits = static_cast<std::uint32_t>(value);
    return bits <= 2147483647U ? static_cast<std::int64_t>(bits)
                             : static_cast<std::int64_t>(bits) - 4294967296LL;
}

// 一次 analyze 调用对应一个 Analysis 实例，以下状态只在此次遍历期间使用。
class Analysis {
    Model &model;
    lexical::Result &words;
    // 作用域编号从 1 开始，访问 scopes 时用编号减 1；0 表示当前不在任何作用域。
    // 深度使用计数而非布尔值，退出内层循环/switch 后仍能正确保留外层状态。
    int current = 0, loopDepth = 0, switchDepth = 0;
    Type returnType{BaseType::Void, false};

    // ===== 考试定位 2：e 严格匹配；Invalid 表示已有错误，不再级联报 e =====
    // map 按行号排序，emplace 不覆盖已有键，因此同一行只保留先记录的错误。
    // 这也保证语义诊断不会覆盖词法/语法阶段已经写入的诊断。
    void error(int line, char code) { words.errors.emplace(line, code); }
    // 统一检查“实际类型”能否用于“期望类型”。Invalid 返回 false，但不再次报 e。
    // 关闭严格模式时仅放宽两个标量之间的匹配，数组与标量仍不能互换。
    bool compatible(Type actual, Type expected, int line) {
        if (!actual.valid() || !expected.valid()) return false;
        if (sameType(actual, expected)) return true;
        if (!SEMANTIC_STRICT_SCALAR_TYPES && actual.scalar() && expected.scalar()) return true;
        error(line, 'e');
        return false;
    }
    // 新建一层：记录旧 current 作为 parent，再将 current 切到新层。
    // 编号按创建顺序递增；有对应语法树节点时，额外保存节点到作用域的映射。
    void enterScope(const Node *block = nullptr) {
        const int id = static_cast<int>(model.scopes.size()) + 1;
        model.scopes.push_back(Scope{id, current, {}, {}});
        current = id;
        if (block) model.blockScopes[block] = id;
    }
    // 离开时仅回到父层，不删除 Scope/Symbol，保证最终输出和绑定仍然有效。
    void leaveScope() { current = model.scopes[current - 1].parent; }
    // 名字查找遵循“最近声明优先”：先本层再父层，允许内层遮蔽外层。
    // 全部查找失败报告 c，返回哨兵，调用方据此传播 Invalid 类型。
    std::size_t lookup(const lexical::Token &name) {
        for (int scope = current; scope; scope = model.scopes[scope - 1].parent) {
            const auto &names = model.scopes[scope - 1].names;
            const auto found = names.find(name.text);
            if (found != names.end()) return found->second;
        }
        error(name.line, 'c');
        return noSymbol;
    }
    // 同层同名才属于重定义 b，不能用 lookup 检查，否则会误拒绝合法遮蔽。
    // 不论是否重定义都分配独立记录；名字表仅绑定第一次声明。
    std::size_t define(Symbol symbol, const Node *node = nullptr) {
        auto &scope = model.scopes[current - 1];
        symbol.duplicate = scope.names.count(symbol.name) != 0;
        if (symbol.duplicate) error(symbol.line, 'b');
        symbol.id = model.symbols.size(); symbol.scope = current; symbol.declaration = node;
        // 重定义保留独立记录，以便分析它自己的初值，但查名字仍得到第一次声明。
        if (!symbol.duplicate) scope.names.emplace(symbol.name, symbol.id);
        scope.symbols.push_back(symbol.id);
        model.symbols.push_back(std::move(symbol));
        if (node) model.bindings[node] = model.symbols.back().id;
        return model.symbols.back().id;
    }
    // 内置函数与普通函数使用相同的查找、形参检查逻辑，仅输出时跳过。
    void builtin(const std::string &name, Type type, std::vector<Type> params) {
        Symbol symbol;
        symbol.name = name; symbol.type = type; symbol.parameters = std::move(params);
        symbol.function = symbol.builtin = true;
        define(std::move(symbol));
    }
    // 缓存节点的完整表达式结果，并同步旧的类型缓存。
    // symbol 有效时还保存名字绑定，后续阶段无需重新按名称解析作用域。
    Expression remember(const Node &node, Expression value) {
        model.expressions[&node] = value;
        model.expressionTypes[&node] = value.type; // 保留旧接口，后续代码可逐步改用 expressions。
        if (value.symbol != noSymbol) model.bindings[&node] = value.symbol;
        return value;
    }

    // ===== 考试定位 3：LVal 的类型、下标和常量数组取值 =====
    // 处理 a 或 a[i]：先查名字，再分析下标，即使名字不存在也检查下标表达式。
    // 下标要求 Int；函数名不能直接作为左值，普通标量也不能使用下标。
    // a 保留数组类型，a[i] 变成元素类型；只对 const 对象尝试读取已知初值。
    // 这里利用已知长度安全取常量值，没有实现数组越界诊断。
    Expression lvalue(const Node &node) {
        const auto *nameNode = terminal(node, "IDENFR");
        const auto &name = *nameNode->token;
        const auto id = lookup(name);
        const auto *index = rule(node, "Exp");
        Expression offset;
        if (index) {
            offset = expression(*index);
            compatible(offset.type, {BaseType::Int, false}, index->firstLine);
        }
        if (id == noSymbol) return remember(node, {});
        model.bindings[nameNode] = id;
        const auto &symbol = model.symbols[id];
        Expression result{symbol.type, std::nullopt, id};
        if (symbol.function) {
            error(name.line, 'e'); result.type = {};
        } else if (index) {
            if (!symbol.type.array) { error(name.line, 'e'); result.type = {}; }
            else {
                result.type.array = false;
                if (symbol.constant && offset.value && *offset.value >= 0) {
                    const auto i = static_cast<std::uint64_t>(*offset.value);
                    if (i < symbol.initial.size()) result.value = symbol.initial[static_cast<std::size_t>(i)];
                    else if (symbol.zeroRemainder && symbol.length && i < static_cast<std::uint64_t>(*symbol.length))
                        result.value = 0;
                }
            }
        } else if (symbol.constant && symbol.type.scalar() && !symbol.initial.empty()) result.value = symbol.initial.front();
        return remember(node, result);
    }

    // 函数名不存在报 c；名字存在但不是函数报 e；数量 d；类型 e。
    // 即使函数查找失败，仍遍历全部实参，尽量发现实参内部的独立错误。
    // 实参数量不符时不再逐项配对，避免越界和无意义的类型比较。
    // 调用的表达式类型取函数返回类型，函数运行结果在此阶段保持未知。
    Expression call(const Node &node) {
        const auto *nameNode = terminal(node, "IDENFR");
        const auto &name = *nameNode->token;
        const auto id = lookup(name); // 与参考相同：先查函数名，再访问实参。
        std::vector<Expression> arguments;
        if (const auto *params = rule(node, "FuncRParams"))
            for (const auto &child : params->children)
                if (child->name == "Exp") arguments.push_back(expression(*child));
        if (id == noSymbol) return remember(node, {});
        model.bindings[nameNode] = id;
        const auto &function = model.symbols[id];
        if (!function.function) { error(name.line, 'e'); return remember(node, {}); }
        if (arguments.size() != function.parameters.size()) error(name.line, 'd');
        else for (std::size_t i = 0; i < arguments.size(); ++i)
            compatible(arguments[i].type, function.parameters[i], name.line);
        return remember(node, {function.type, std::nullopt, id});
    }

    // ===== 考试定位 4：表达式类型与常量折叠 =====
    // UnaryExp 分三种：一元运算、显式类型转换、函数调用或普通初等表达式。
    // 一元运算只接受有效标量；! 结果为 Int，其余此处保留操作数类型。
    // ++/-- 检查常量修改 h 并清除已知值，不在语义阶段模拟实际执行。
    Expression unary(const Node &node) {
        if (const auto *opNode = rule(node, "UnaryOp")) {
            const auto &op = *token(*opNode);
            const auto *operand = rule(node, "UnaryExp");
            if (!operand) operand = rule(node, "LVal"); // 兼容此前的 ++/-- 语法练习。
            auto value = expression(*operand);
            if (value.type.valid() && !value.type.scalar()) { error(op.line, 'e'); return remember(node, {}); }
            if (!value.type.valid()) return remember(node, {});
            const auto resultType = op.text == "!" ? Type{BaseType::Int, false} : value.type;
            if (op.text == "++" || op.text == "--") {
                if (value.symbol != noSymbol && model.symbols[value.symbol].constant) error(op.line, 'h');
                value.value.reset();
            } else if (value.value) {
                if (op.text == "!") value.value = !*value.value;
                else if (op.text == "-") value.value = narrow(-*value.value, resultType);
                else if (op.text == "~") value.value = narrow(~*value.value, resultType);
            }
            // 关键区别：参考代码里 +char、-char 仍是 Char，不自动提升。
            return remember(node, {resultType, value.value, noSymbol});
        }
        // 显式转换先检查操作数是标量，再按目标类型截断已知常量值。
        if (const auto *target = rule(node, "BType")) {
            auto value = expression(*rule(node, "UnaryExp"));
            if (!value.type.valid()) return remember(node, {});
            if (!value.type.scalar()) { error(node.firstLine, 'e'); return remember(node, {}); }
            const auto type = declaredType(*target);
            return remember(node, {type, value.value ? std::optional<std::int64_t>(narrow(*value.value, type)) : std::nullopt, noSymbol});
        }
        if (terminal(node, "IDENFR")) return call(node);
        return remember(node, expression(*node.children.front()));
    }

    // 二元文法节点可能只有一个孩子（没有真正的运算），直接向上传递结果。
    // 真正的运算形状为 [左表达式, 运算符, 右表达式]，递归分析两侧。
    // 本函数用于静态检查：逻辑运算也访问两边，不模拟运行时短路执行。
    Expression binary(const Node &node) {
        if (node.children.size() == 1) return remember(node, expression(*node.children.front()));
        const auto left = expression(*node.children[0]), right = expression(*node.children[2]);
        const auto &op = *node.children[1]->token;
        if (!left.type.valid() || !right.type.valid()) return remember(node, {});
        if (!left.type.scalar() || !right.type.scalar() ||
            (SEMANTIC_STRICT_SCALAR_TYPES && !sameType(left.type, right.type))) {
            error(op.line, 'e'); return remember(node, {});
        }
        // 比较和逻辑运算得到 Int；普通算术在这里保留左操作数类型。
        // 严格模式已确保左右基本类型一致，且都不是数组或 Void。
        const bool comparison = node.name == "RelExp" || node.name == "EqExp" ||
                                node.name == "LAndExp" || node.name == "LOrExp";
        Expression result{comparison ? Type{BaseType::Int, false} : left.type, std::nullopt, noSymbol};
        // optional 的布尔判断表示“有值”，所以已知的 0 也会参与折叠。
        // 仅两侧都已知时计算；除数为零则跳过除法/取模，保持 value 为空。
        if (left.value && right.value) {
            const auto a = *left.value, b = *right.value;
            const auto &s = op.text;
            // a/b 已在 int32/uint8 范围，用 int64 计算避免宿主 int 溢出。
            if (s == "+") result.value = narrow(a + b, result.type);
            else if (s == "-") result.value = narrow(a - b, result.type);
            else if (s == "*") result.value = narrow(a * b, result.type);
            else if (s == "/" && b) result.value = narrow(a / b, result.type);
            else if (s == "%" && b) result.value = narrow(a % b, result.type);
            else if (s == "<") result.value = a < b;
            else if (s == ">") result.value = a > b;
            else if (s == "<=") result.value = a <= b;
            else if (s == ">=") result.value = a >= b;
            else if (s == "==") result.value = a == b;
            else if (s == "!=") result.value = a != b;
            else if (s == "&&" || (s == "&" && node.name == "LAndExp")) result.value = a && b;
            else if (s == "||" || (s == "|" && node.name == "LOrExp")) result.value = a || b;
            // 预测题位运算只作有限扩展；课程默认开关关闭。
            else if (s == "&") result.value = narrow(a & b, result.type);
            else if (s == "|") result.value = narrow(a | b, result.type);
            else if (s == "^") result.value = narrow(a ^ b, result.type);
        }
        return remember(node, result); // 除零等无法安全折叠时保留“未知”，不新增错误码。
    }

    // 表达式统一入口：按文法节点名分派，所有路径尽量通过 remember 保存结果。
    // Exp/ConstExp 等包装节点不新增运算，直接继承孩子的语义信息。
    Expression expression(const Node &node) {
        if (node.name == "LVal") return lvalue(node);
        if (node.name == "Number") {
            const auto &word = *token(node);
            if (word.kind == "FLOATCON") throw std::runtime_error("floating-point semantics are outside this assignment");
            const Type type{word.kind == "CHARCON" ? BaseType::Char : BaseType::Int, false};
            return remember(node, {type, word.number ? std::optional<std::int64_t>(narrow(*word.number, type)) : std::nullopt, noSymbol});
        }
        if (node.name == "UnaryExp") return unary(node);
        // PrimaryExp 的首孩子若是终结符（如左括号），实际表达式位于第二个孩子；
        // 否则首孩子就是 LVal/Number 等。括号本身不改变表达式类型。
        if (node.name == "PrimaryExp") {
            auto value = expression(*node.children[node.children.front()->token ? 1 : 0]);
            if (token(node, "++") || token(node, "--")) {
                if (value.symbol != noSymbol && model.symbols[value.symbol].constant) error(node.firstLine, 'h');
                value.value.reset();
            }
            return remember(node, value);
        }
        // 条件要求标量；已知值归一化为 0/1，有效条件的结果类型统一记为 Int。
        if (node.name == "Cond") {
            auto value = expression(*node.children.front());
            if (value.type.valid() && !value.type.scalar()) { error(node.firstLine, 'e'); value.type = {}; }
            if (value.value) value.value = *value.value != 0;
            return remember(node, {value.type.valid() ? Type{BaseType::Int, false} : Type{}, value.value, noSymbol});
        }
        if (node.name == "Exp" || node.name == "ConstExp") return remember(node, expression(*node.children.front()));
        return binary(node);
    }

    // ===== 考试定位 5：数组长度、初值、补零与 const 可求值信息 =====
    // 初值按元素类型检查：数组去掉 array 标记后，与每一项初值匹配。
    // 字符串使用词法阶段已解码的字节，本函数只记录这些字节，不额外追加字符。
    // 普通表达式记录可选常量值；未能求值仍保留该项的“未知”状态。
    // 用 id 每次重新访问符号，避免递归处理时依赖可能失效的容器元素指针。
    void initializer(const Node &node, std::size_t id) {
        Type expected = model.symbols[id].type; expected.array = false;
        for (const auto &child : node.children) {
            if (child->token && child->token->kind == "STRCON") {
                compatible({BaseType::Char, false}, expected, child->firstLine);
                for (const unsigned char byte : child->token->decoded) model.symbols[id].initial.push_back(byte);
            } else if (!child->token) {
                const auto value = expression(*child);
                compatible(value.type, expected, child->firstLine);
                model.symbols[id].initial.push_back(value.value);
            }
        }
        model.symbols[id].zeroRemainder = true;
    }
    // 一个声明组可包含多个定义，例如 int a, b[3]；逐项建立 Symbol。
    // 处理顺序：数组长度表达式 -> 登记当前符号 -> 分析初始化表达式。
    // 因此长度计算时当前符号尚未入表，初始化时已能查到当前名字。
    void declaration(const Node &decl) {
        const Node *group = rule(decl, "ConstDecl");
        const bool constant = group != nullptr;
        if (!group) group = rule(decl, "VarDecl");
        const auto base = declaredType(*rule(*group, "BType"));
        for (const auto &child : group->children) {
            if (child->name != "ConstDef" && child->name != "VarDef") continue;
            const auto *nameNode = terminal(*child, "IDENFR");
            const auto &name = *nameNode->token;
            Symbol symbol;
            symbol.name = name.text; symbol.line = name.line; symbol.type = base;
            symbol.type.array = token(*child, "[") != nullptr;
            symbol.constant = constant; symbol.staticStorage = token(*group, "static") != nullptr;
            if (const auto *length = rule(*child, "ConstExp")) {
                const auto size = expression(*length);
                compatible(size.type, {BaseType::Int, false}, length->firstLine);
                symbol.length = size.value;
            }
            // 无显式初值时，记录全局/static/const 对象的补零属性。
            // 有显式初始化则 initializer 会再设置此标记；这里不展开整个数组。
            symbol.zeroRemainder = current == 1 || symbol.staticStorage || constant;
            const auto id = define(std::move(symbol), child.get());
            model.bindings[nameNode] = id;
            if (const auto *init = rule(*child, constant ? "ConstInitVal" : "InitVal")) initializer(*init, id);
        }
    }
    // 按 BlockItem 的源码顺序访问，声明在入表后才对后面的语句可见。
    // 普通嵌套块建立新层；函数体传 false，复用已经放入形参的那一层。
    void block(const Node &node, bool newScope) {
        if (newScope) enterScope(&node);
        for (const auto &item : node.children) {
            if (item->name != "BlockItem") continue;
            for (const auto &child : item->children) {
                if (child->name == "Decl") declaration(*child);
                else if (child->name == "Stmt") statement(*child);
            }
        }
        if (newScope) leaveScope();
    }

    // ===== 考试定位 6：形参与函数最外层共用作用域；return; 的 e 与缺 return 的 g 分开 =====
    // 普通函数先在外层登记返回类型与完整形参签名，再分析函数体，支持自递归。
    // main 按 Int 返回类型分析，但此实现不把 main 登记为输出函数符号。
    // 签名中的形参类型用于调用检查，函数层中的形参 Symbol 用于体内名字查找。
    void function(const Node &node) {
        const bool main = node.name == "MainFuncDef";
        const Type result = main ? Type{BaseType::Int, false} : declaredType(*rule(node, "FuncType"));
        const auto *params = rule(node, "FuncFParams");
        const auto *body = rule(node, "Block");
        if (!main) {
            const auto *nameNode = terminal(node, "IDENFR");
            Symbol symbol;
            symbol.name = nameNode->token->text; symbol.line = nameNode->token->line;
            symbol.function = true; symbol.type = result;
            if (params) for (const auto &param : params->children) {
                if (param->name != "FuncFParam") continue;
                auto type = declaredType(*rule(*param, "BType")); type.array = token(*param, "[") != nullptr;
                symbol.parameters.push_back(type);
            }
            const auto id = define(std::move(symbol), &node);
            model.bindings[nameNode] = id;
        }
        returnType = result;
        enterScope(body);
        if (params) for (const auto &param : params->children) {
            if (param->name != "FuncFParam") continue;
            const auto *nameNode = terminal(*param, "IDENFR");
            Symbol parameter;
            parameter.name = nameNode->token->text; parameter.line = nameNode->token->line;
            parameter.type = declaredType(*rule(*param, "BType"));
            parameter.type.array = token(*param, "[") != nullptr; parameter.parameter = true;
            const auto id = define(std::move(parameter), param.get());
            model.bindings[nameNode] = id;
        }
        // 复用函数层：形参与最外层局部变量同名会触发 b，嵌套块则可遮蔽。
        block(*body, false);
        // g 只检查末尾是否直接为 return，不做控制流路径分析。
        // 若最后一个 BlockItem 是声明，last 会变成 nullptr，同样视为不满足要求。
        // 非 void 的 return; 由 statement 报 e，不能把它与缺少末尾 return 混为一谈。
        const Node *last = nullptr;
        for (const auto &item : body->children)
            if (item->name == "BlockItem") last = rule(*item, "Stmt");
        // 参考代码只要求末尾直接是 return 语句，不要求它必有 Exp。
        if (result.base != BaseType::Void && (!last || !token(*last, "return"))) error(body->lastLine, 'g');
        leaveScope(); returnType = {BaseType::Void, false};
    }

    // ===== 考试定位 7：使用解码后的格式串，%d/%c/%s 分别要求 Int/Char/CharArray =====
    // 根据格式串构造有序的期望参数列表，再分析全部实际参数。
    // %% 不消耗参数，%d/%c/%s 分别消耗 Int/Char/CharArray。
    // 数量不一致报 l；数量相同时才逐项调用 compatible，类型不符报 e。
    void printfStatement(const Node &node) {
        const auto &format = terminal(node, "STRCON")->token->decoded;
        std::vector<Type> expected;
        for (std::size_t i = 0; i + 1 < format.size(); ++i) {
            if (format[i] != '%') continue;
            const char c = format[++i]; // %% 被跳过，不是占位符。
            if (c == 'd') expected.push_back({BaseType::Int, false});
            else if (c == 'c') expected.push_back({BaseType::Char, false});
            else if (c == 's') expected.push_back({BaseType::Char, true});
        }
        std::vector<Expression> arguments;
        for (const auto &child : node.children)
            if (child->name == "Exp") arguments.push_back(expression(*child));
        const int line = token(node, "printf")->line;
        if (arguments.size() != expected.size()) error(line, 'l');
        else for (std::size_t i = 0; i < arguments.size(); ++i) compatible(arguments[i].type, expected[i], line);
    }

    // ===== 考试定位 8：case 类型 e；重复标签 n；case 行号取关键字而不是 Number =====
    // 控制表达式要求标量，case 类型需与控制表达式兼容。
    // labels 仅属于当前 switch，内外 switch 的相同标签互不冲突。
    // 标签数值用 set 去重，default 用布尔标记；错误行号取 case/default 关键字。
    // switchDepth 只让 break 合法，不会让 switch 外无循环的 continue 合法。
    void analyzeSwitch(const Node &node) {
        auto control = expression(*rule(node, "Exp"));
        if (control.type.valid() && !control.type.scalar()) { error(node.firstLine, 'e'); control.type = {}; }
        if (SEMANTIC_SWITCH_SCOPE) enterScope(&node);
        ++switchDepth;
        std::set<std::int64_t> labels;
        bool defaultSeen = false;
        for (const auto &child : node.children) {
            if (child->name != "CaseStmt") continue;
            const int line = child->children.front()->firstLine;
            if (const auto *number = rule(*child, "Number")) {
                const auto value = expression(*number);
                compatible(value.type, control.type, line);
                if (SEMANTIC_CHECK_DUPLICATE_CASE && value.value && !labels.insert(*value.value).second) error(line, 'n');
            } else {
                if (SEMANTIC_CHECK_DUPLICATE_CASE && defaultSeen) error(line, 'n');
                defaultSeen = true;
            }
            for (const auto &stmt : child->children) if (stmt->name == "Stmt") statement(*stmt);
        }
        --switchDepth;
        if (SEMANTIC_SWITCH_SCOPE) leaveScope();
    }

    // 先处理拥有专门规则的语句并返回，再通用遍历条件、赋值和嵌套语句。
    // if/else 等结构无需额外分支：其 Exp/Cond/Stmt 子节点由末尾循环递归检查。
    void statement(const Node &node) {
        if (const auto *body = rule(node, "Block")) { block(*body, true); return; }
        if (token(node, "switch")) { analyzeSwitch(node); return; }
        if (const auto *keyword = token(node, "break")) {
            if (!loopDepth && !switchDepth) error(keyword->line, 'm');
            return;
        }
        if (const auto *keyword = token(node, "continue")) {
            if (!loopDepth) error(keyword->line, 'm');
            return;
        }
        if (const auto *keyword = token(node, "return")) {
            const auto *value = rule(node, "Exp");
            const auto actual = value ? expression(*value) : Expression{{BaseType::Void, false}, std::nullopt, noSymbol};
            if (returnType.base == BaseType::Void) { if (value) error(keyword->line, 'f'); }
            else compatible(actual.type, returnType, keyword->line);
            return;
        }
        if (token(node, "printf")) { printfStatement(node); return; }
        // for 单独建立作用域，使其声明的循环变量在退出循环后不再可见。
        // 进入循环前递增深度，处理完所有孩子后恢复，支持多层嵌套。
        const bool forLoop = token(node, "for") != nullptr;
        const bool looping = forLoop || token(node, "while") || token(node, "do") || token(node, "repeat");
        if (forLoop) enterScope(&node);
        if (looping) ++loopDepth;
        for (std::size_t i = 0; i < node.children.size(); ++i) {
            const auto &child = *node.children[i];
            if (child.name == "Decl") declaration(child);
            else if (child.name == "LVal" && i + 2 < node.children.size() && node.children[i + 2]->name == "Exp") {
                // Parser 将赋值组织为 LVal、赋值符、Exp 三个相邻孩子。
                // 左右两侧都分析后，检查 const 修改和类型，禁止整个数组直接赋值。
                const auto left = expression(child), right = expression(*node.children[i + 2]);
                if (left.symbol != noSymbol && model.symbols[left.symbol].constant) error(child.firstLine, 'h');
                if (left.type.valid() && left.type.array) error(child.firstLine, 'e');
                else compatible(right.type, left.type, child.firstLine);
                // 右侧 Exp 已处理，跳过赋值符与右侧节点，避免重复诊断和记录。
                i += 2;
            } else if (child.name == "Exp" || child.name == "Cond") expression(child);
            else if (child.name == "Stmt") statement(child);
        }
        if (looping) --loopDepth;
        if (forLoop) leaveScope();
    }

  public:
    Analysis(Model &output, lexical::Result &input) : model(output), words(input) {}
    // 建立全局层（编号 1），预登记运行时函数，再按顺序处理顶层声明/函数。
    // 不预扫描后续函数，因此自身递归可用，但尚未登记的后续函数无法查到。
    void run(const Node &tree) {
        enterScope();
        const Type integer{BaseType::Int, false}, character{BaseType::Char, false};
        const Type characters{BaseType::Char, true}, nothing{BaseType::Void, false};
        builtin("get_int", integer, {}); builtin("get_char", character, {});
        builtin("get_string", nothing, {characters, integer});
        builtin("put_int", nothing, {integer}); builtin("put_char", nothing, {character});
        builtin("put_string", nothing, {characters}); builtin("put_str", nothing, {characters});
        for (const auto &child : tree.children) {
            if (child->name == "Decl") declaration(*child);
            else if (child->name == "FuncDef" || child->name == "MainFuncDef") function(*child);
        }
        leaveScope();
    }
};
} // namespace

// 将符号属性转换为课程要求的输出名称，例如 ConstIntArray、CharFunc。
// 输出只表达类型类别，不输出形参列表、初值或常量折叠结果。
std::string typeName(const Symbol &symbol) {
    const std::string base = symbol.type.base == BaseType::Int ? "Int" : symbol.type.base == BaseType::Char ? "Char" : "Void";
    if (symbol.function) return base + "Func";
    return std::string(symbol.constant ? "Const" : symbol.staticStorage ? "Static" : "") + base + (symbol.type.array ? "Array" : "");
}
// scopes 的创建顺序决定层次编号，每层 symbols 的顺序决定该层声明顺序。
// 不遍历 unordered_map 输出，避免哈希表顺序不稳定。内置和重定义记录跳过。
// 是否存在错误由 driver 决定；本函数本身不检查 words.errors。
void writeSymbols(const Model &model, std::ostream &out) {
    for (const auto &scope : model.scopes)
        for (const auto id : scope.symbols) {
            const auto &symbol = model.symbols[id];
            if (!symbol.builtin && !symbol.duplicate) out << scope.id << ' ' << symbol.name << ' ' << typeName(symbol) << '\n';
        }
}
// 对外入口：创建独立结果对象，并把共享错误表借给本次遍历使用。
// 返回后 Model 持有结果数据，但其中 Node* 不拥有语法树，调用方须保留 tree。
std::unique_ptr<Model> SemanticAnalyzer::analyze(const syntax::Node &tree, lexical::Result &words) {
    auto model = std::make_unique<Model>();
    Analysis(*model, words).run(tree);
    return model;
}
} // namespace semantic
