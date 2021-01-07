#include <memory>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

#include <jit/jit.h>

// forward declaration for Visitor
class BinaryExprAST;
class UnaryExprAST;
class NumberExprAST;
class IdentifierExprAST;

// visitor interface
class Visitor {
    public:
        virtual void visit_binary_node(const BinaryExprAST *node) = 0;
        virtual void visit_unary_node(const UnaryExprAST *node) = 0;
        virtual void visit_number_node(const NumberExprAST *node) = 0;
        virtual void visit_identifier_node(const IdentifierExprAST *node) = 0;
};

// AST nodes
struct ExprAST {
    virtual ~ExprAST() = default;
    virtual void accept(Visitor *visitor) const = 0;
};

// AST node for raw values, only double is supported
struct NumberExprAST: public ExprAST {
    const jit_float64 value;

    NumberExprAST(jit_float64 value): value{value} {}
    void accept(Visitor *visitor) const { visitor->visit_number_node(this); }
};

// AST node for identifier
struct IdentifierExprAST: public ExprAST {
    const std::string identifier;
    IdentifierExprAST(const std::string &id): identifier{id} {}
    void accept(Visitor *visitor) const { visitor->visit_identifier_node(this); }
};

enum class BinaryOperator {
    Plus,
    Minus,
    Mult,
    Div,
};

// AST node that represents binary operations listed in BinaryOperator enum
struct BinaryExprAST: public ExprAST {
    const BinaryOperator op;
    const std::unique_ptr<ExprAST> lhs, rhs;
    BinaryExprAST(BinaryOperator op, std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs):
        op{op}, lhs{std::move(lhs)}, rhs{std::move(rhs)} {}
    void accept(Visitor *visit) const { visit->visit_binary_node(this); }
};

enum class UnaryOperator {
    Acos,
    Asin,
    Atan,
    Cos,
    Cosh,
    Exp,
    Log10,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
};

// AST node that represents unary operations listed in UnaryOperator enum
struct UnaryExprAST: public ExprAST {
    UnaryOperator op;
    std::unique_ptr<ExprAST> arg;

    UnaryExprAST(UnaryOperator op, std::unique_ptr<ExprAST> arg):
        op{op}, arg{std::move(arg)} {}
    void accept(Visitor *visit) const { visit->visit_unary_node(this); }
};

//Visitor implementations for codegen and AST analysis
class CodegenVisitor: public Visitor {
    jit_function_t function;
    const std::unordered_map<std::string, jit_float64> identifier_map;
    jit_value_t current_result;

    public:
        CodegenVisitor(jit_function_t &function, const std::unordered_map<std::string, jit_float64> &identifier_map):
            function{function}, identifier_map{identifier_map} {}

        void compile(ExprAST const &ast) {
            ast.accept(this);
            // set return value
            jit_insn_return(function, current_result);
        }

        void visit_binary_node(const BinaryExprAST *node) {
            node->lhs->accept(this);
            jit_value_t tmp_left = current_result;
            node->rhs->accept(this);
            jit_value_t tmp_right = current_result;

            switch(node->op) {
                case BinaryOperator::Plus:
                    current_result = jit_insn_add(function, tmp_left, tmp_right);
                    break;
                case BinaryOperator::Mult:
                    current_result = jit_insn_mul(function, tmp_left, tmp_right);
                    break;
                case BinaryOperator::Minus:
                    current_result = jit_insn_sub(function, tmp_left, tmp_right);
                    break;
                case BinaryOperator::Div:
                    current_result = jit_insn_div(function, tmp_left, tmp_right);
                    break;
            }
        }

        void visit_unary_node(const UnaryExprAST *node) {
            node->arg->accept(this);
            jit_value_t tmp = current_result;

            switch(node->op) {
                case UnaryOperator::Acos:
                    current_result = jit_insn_acos(function, tmp);
                    break;
                case UnaryOperator::Asin:
                    current_result = jit_insn_asin(function, tmp);
                    break;
                case UnaryOperator::Atan:
                    current_result = jit_insn_atan(function, tmp);
                    break;
                case UnaryOperator::Cos:
                    current_result = jit_insn_cos(function, tmp);
                    break;
                case UnaryOperator::Cosh:
                    current_result = jit_insn_cosh(function, tmp);
                    break;
                case UnaryOperator::Exp:
                    current_result = jit_insn_exp(function, tmp);
                    break;
                case UnaryOperator::Log10:
                    current_result = jit_insn_log10(function, tmp);
                    break;
                case UnaryOperator::Sin:
                    current_result = jit_insn_sin(function, tmp);
                    break;
                case UnaryOperator::Sinh:
                    current_result = jit_insn_sinh(function, tmp);
                    break;
                case UnaryOperator::Sqrt:
                    current_result = jit_insn_sqrt(function, tmp);
                    break;
                case UnaryOperator::Tan:
                    current_result = jit_insn_tan(function, tmp);
                    break;
                case UnaryOperator::Tanh:
                    current_result = jit_insn_tanh(function, tmp);
                    break;
            }
        }

        void visit_number_node(const NumberExprAST *node) {
            current_result = jit_value_create_float64_constant(function, jit_type_float64, node->value);
        }

        void visit_identifier_node(const IdentifierExprAST *node) {
            assert(identifier_map.find(node->identifier) != identifier_map.end());
            int arg_index = std::distance(identifier_map.begin(), identifier_map.find(node->identifier));
            current_result = jit_value_get_param(function, arg_index);
        }
};

// using the C version of API
void compile_and_run(ExprAST const &ast, std::unordered_map<std::string, jit_float64> &identifier_map)
{
    jit_context_t context;
    jit_type_t signature;

    context = jit_context_create();
    jit_context_build_start(context);
    {
        std::vector<jit_type_t> params(identifier_map.size());
        std::fill(params.begin(), params.end(), jit_type_float64);
        signature = jit_type_create_signature(jit_abi_cdecl, jit_type_float64, params.data(), params.size(), 1);
        jit_function_t function = jit_function_create(context, signature);
        jit_type_free(signature);

        CodegenVisitor cv(function, identifier_map);
        cv.compile(ast);
        jit_function_compile(function);

        jit_float64 result;
        //Dumb API
        std::vector<void*> args(identifier_map.size());
        std::transform(identifier_map.begin(), identifier_map.end(), args.begin(), [](auto &p) { return &p.second; });
        jit_function_apply(function, args.data(), &result);
        printf("Result: %lf\n", result);
    }
    jit_context_build_end(context);
    jit_context_destroy(context);
}

//
// Helper functions to make this look more concise, one could also use operator overload
//

std::unique_ptr<NumberExprAST> Number(double val)
{
    return std::make_unique<NumberExprAST>(static_cast<jit_float64>(val));
}

std::unique_ptr<IdentifierExprAST> Identifier(std::string identifier) {
    return std::make_unique<IdentifierExprAST>(identifier);
}

std::unique_ptr<BinaryExprAST> Mult(std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs)
{
    return std::make_unique<BinaryExprAST>(BinaryOperator::Mult, std::move(lhs), std::move(rhs));
}

std::unique_ptr<BinaryExprAST> Add(std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs)
{
    return std::make_unique<BinaryExprAST>(BinaryOperator::Plus, std::move(lhs), std::move(rhs));
}

int main(void)
{
    auto ast = Add(Mult(Number(1), Number(2)), Mult(Identifier("y"), Identifier("x")));
    std::unordered_map<std::string, jit_float64> args = {{"x", 6}, {"y", 2}};
    compile_and_run(*ast, args);
}
