import ast
import argparse
import matplotlib.pyplot as plt

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    @property
    def depth(self):
        left_d = self.left.depth if self.left else 0
        right_d = self.right.depth if self.right else 0
        return 1 + max(left_d, right_d)

def ast_to_node(ast_node):
    if isinstance(ast_node, ast.Call):
        func_name = ast_node.func.id if isinstance(ast_node.func, ast.Name) else str(ast_node.func)
        if len(ast_node.args) == 2:
            return Node(func_name, ast_to_node(ast_node.args[0]), ast_to_node(ast_node.args[1]))
        elif len(ast_node.args) == 1:
            return Node(func_name, ast_to_node(ast_node.args[0]))
        else:
            return Node(func_name)
    elif isinstance(ast_node, ast.BinOp):
        op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
        }
        op = op_map.get(type(ast_node.op), str(type(ast_node.op)))
        return Node(op, ast_to_node(ast_node.left), ast_to_node(ast_node.right))
    elif isinstance(ast_node, ast.Name):
        name = ast_node.id
        if name.startswith('load_t_'):
            name = name.replace('load_t_', 'load_t-')
        return Node(name)
    elif isinstance(ast_node, ast.Constant):
        return Node(str(ast_node.value))
    else:
        return Node(type(ast_node).__name__)

def plot_tree_string(expr_str, output_filename="tree_plot.png"):
    safe_expr = expr_str.replace('load_t-', 'load_t_')
    
    try:
        tree_ast = ast.parse(safe_expr, mode='eval').body
        root = ast_to_node(tree_ast)
    except SyntaxError as e:
        print(f"Failed to parse expression: {e}")
        return

    depth = root.depth
    width = min(100, max(10, (2 ** depth) * 0.8))
    height = min(50, max(6, (depth + 1) * 1.5))
    
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    
    def draw_node(node, x, y, dx, dy):
        if node.left and node.right:
            ax.plot([x, x - dx], [y, y - dy], 'k-', lw=1.5, zorder=1)
            ax.plot([x, x + dx], [y, y - dy], 'k-', lw=1.5, zorder=1)
            draw_node(node.left, x - dx, y - dy, dx / 2, dy)
            draw_node(node.right, x + dx, y - dy, dx / 2, dy)
        elif node.left:
            ax.plot([x, x], [y, y - dy], 'k-', lw=1.5, zorder=1)
            draw_node(node.left, x, y - dy, dx / 2, dy)
            
        ax.text(x, y, str(node.value), ha='center', va='center',
            bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1'),
            fontsize=12,
            zorder=2
        )

    draw_node(root, 0, 0, width / 2, 1)
    
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved tree visualization to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a tree from a string expression")
    parser.add_argument("expression", type=str, help="The expression string to plot")
    parser.add_argument("--output", type=str, default="tree_plot.png", help="Output image filename")
    
    args = parser.parse_args()
    plot_tree_string(args.expression, args.output)
