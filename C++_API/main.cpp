#include <memory>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

#include <jit/jit-plus.h>

// forward declaration for Visitor
struct BinaryExprAST;
struct UnaryExprAST;
struct NumberExprAST;
struct IdentifierExprAST;

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

    explicit NumberExprAST(jit_float64 value): value{value} {}
    void accept(Visitor *visitor) const override { visitor->visit_number_node(this); }
};

// AST node for identifier
struct IdentifierExprAST: public ExprAST {
    const std::string &identifier;
    explicit IdentifierExprAST(const std::string &id): identifier{id} {}
    void accept(Visitor *visitor) const override { visitor->visit_identifier_node(this); }
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
    void accept(Visitor *visit) const override { visit->visit_binary_node(this); }
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
    void accept(Visitor *visit) const override { visit->visit_unary_node(this); }
};

//Visitor implementations for codegen and AST analysis
class UserFunction: public jit_function, public Visitor {
    const ExprAST &ast;
    const std::vector<std::string> &identifiers;
    jit_value current_result;

    public:
        UserFunction(jit_context &context, ExprAST const &ast, const std::vector<std::string> &identifiers):
            jit_function(context), ast{ast}, identifiers{identifiers}
        {
            create();
        }

        jit_type_t create_signature() override
        {
            std::vector<jit_type_t> params(identifiers.size(), jit_type_float64);
            return jit_type_create_signature(jit_abi_cdecl, jit_type_float64, params.data(), params.size(), 1);
        }

        void build() override
        {
            ast.accept(this);
            insn_return(current_result);
        }

        jit_float64 call(std::vector<jit_float64> arguments)
        {
            jit_float64 result;
            std::vector<void*> _args(arguments.size());
            std::transform(arguments.begin(), arguments.end(), _args.begin(), [](auto &i) { return &i; });

            apply(_args.data(), &result);

            return result;
        }

        void visit_binary_node(const BinaryExprAST *node) override
        {
            node->lhs->accept(this);
            jit_value tmp_left = current_result;
            node->rhs->accept(this);
            jit_value tmp_right = current_result;

            switch(node->op) {
                case BinaryOperator::Plus:
                    current_result = insn_add(tmp_left, tmp_right);
                    break;
                case BinaryOperator::Mult:
                    current_result = insn_mul(tmp_left, tmp_right);
                    break;
                case BinaryOperator::Minus:
                    current_result = insn_sub(tmp_left, tmp_right);
                    break;
                case BinaryOperator::Div:
                    current_result = insn_div(tmp_left, tmp_right);
                    break;
            }
        }

        void visit_unary_node(const UnaryExprAST *node) override
        {
            node->arg->accept(this);
            jit_value tmp = current_result;

            switch(node->op) {
                case UnaryOperator::Acos:
                    current_result = insn_acos(tmp);
                    break;
                case UnaryOperator::Asin:
                    current_result = insn_asin(tmp);
                    break;
                case UnaryOperator::Atan:
                    current_result = insn_atan(tmp);
                    break;
                case UnaryOperator::Cos:
                    current_result = insn_cos(tmp);
                    break;
                case UnaryOperator::Cosh:
                    current_result = insn_cosh(tmp);
                    break;
                case UnaryOperator::Exp:
                    current_result = insn_exp(tmp);
                    break;
                case UnaryOperator::Log10:
                    current_result = insn_log10(tmp);
                    break;
                case UnaryOperator::Sin:
                    current_result = insn_sin(tmp);
                    break;
                case UnaryOperator::Sinh:
                    current_result = insn_sinh(tmp);
                    break;
                case UnaryOperator::Sqrt:
                    current_result = insn_sqrt(tmp);
                    break;
                case UnaryOperator::Tan:
                    current_result = insn_tan(tmp);
                    break;
                case UnaryOperator::Tanh:
                    current_result = insn_tanh(tmp);
                    break;
            }
        }

        void visit_number_node(const NumberExprAST *node) override
        {
            current_result = new_constant(node->value, jit_type_float64);
        }

        void visit_identifier_node(const IdentifierExprAST *node) override
        {
            auto it = std::find(identifiers.cbegin(), identifiers.cend(), node->identifier);
            current_result = get_param(std::distance(identifiers.cbegin(), it));
        }
};

//
// Helper functions to make this look more concise, one could also use operator overload
//

std::unique_ptr<NumberExprAST> Number(double val)
{
    return std::make_unique<NumberExprAST>(static_cast<jit_float64>(val));
}

std::unique_ptr<IdentifierExprAST> Identifier(const std::string &identifier) {
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

int main()
{
    auto ast = Add(Mult(Number(1), Number(2)), Mult(Identifier("y"), Identifier("x")));
    std::vector<std::string> identifiers({"x", "y"});

    jit_context context;

    UserFunction f(context, *ast, identifiers);
    std::vector<double> args({3, 5});
    printf("Result: %lf\n", f.call(args));
}
