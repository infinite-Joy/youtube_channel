from manim import *
import numpy as np
from numpy.linalg import cholesky

class GPTQAlgorithm(Scene):
    def construct(self):
        self.setup_scene()
        self.explain_problem()
        self.show_algorithm_overview()
        self.demonstrate_quantization()
        self.visualize_error_redistribution()
        self.show_algorithm_steps()
        self.conclusion()

    def setup_scene(self):
        # Title
        title = Text("GPTQ: Quantization Algorithm", font_size=48)
        subtitle = Text("Post-Training Quantization Method", font_size=36)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(1)
        self.play(
            title.animate.scale(0.6).to_corner(UP + LEFT),
            FadeOut(subtitle)
        )
        
    def explain_problem(self):
        # Explain the basic problem
        problem_text = Tex(
            r"Problem: Quantize weights $W$ into $\hat{W}$ to minimize error",
            font_size=32
        )
        problem_text.to_edge(UP).shift([0,-1.5,0])
        
        equation = MathTex(
            r"\underset{\hat{W}}{\text{argmin}} \, \|WX - \hat{W}X\|^2_2",
            font_size=42
        )
        equation.next_to(problem_text, DOWN, buff=0.5)
        
        explanation = Tex(
            r"Where $X$ is a small calibration dataset of activations",
            font_size=32
        )
        explanation.next_to(equation, DOWN, buff=0.5)
        
        self.play(Write(problem_text))
        self.play(Write(equation))
        self.play(Write(explanation))
        self.wait(2)
        
        self.play(
            FadeOut(problem_text),
            FadeOut(equation), 
            FadeOut(explanation)
        )
        
    def show_algorithm_overview(self):
        # Explain key insights of GPTQ
        insights = VGroup(
            Text("Key Insights of GPTQ:", font_size=36),
            BulletedList(
                "Processes weights in order of importance",
                "Uses Hessian matrix for error redistribution",
                "Quantizes column-by-column in blocks",
                "Redistributes quantization errors to minimize impact",
                font_size=32
            )
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        insights.to_edge(LEFT)
        
        self.play(Write(insights[0]))
        self.play(Write(insights[1]))
        self.wait(2)
        self.play(FadeOut(insights))
        
    def demonstrate_quantization(self):
        # Show how quantization works
        original_weight_values = np.array([0.13, -0.27, 0.51, 0.89, -0.45, 0.32, -0.78, 0.21])
        quantized_weight_values = np.array([0.125, -0.25, 0.5, 0.875, -0.5, 0.375, -0.75, 0.25])
        
        # Create visualization of weights
        original_weights = self.create_weight_visualization(
            original_weight_values, 
            title="Original Weights (32-bit)"
        )
        quantized_weights = self.create_weight_visualization(
            quantized_weight_values, 
            title="Quantized Weights (4-bit)"
        )
        
        original_weights.to_edge(UP).shift([0,-0.5,0])
        quantized_weights.next_to(original_weights, DOWN, buff=0.5)
        
        # Show error
        errors = original_weight_values - quantized_weight_values
        error_viz = self.create_weight_visualization(
            errors, 
            title="Quantization Error",
            value_color=YELLOW
        )
        error_viz.next_to(quantized_weights, DOWN, buff=0.5)
        
        self.play(Write(original_weights))
        self.wait(1)
        self.play(Write(quantized_weights))
        self.wait(1)
        self.play(Write(error_viz))
        self.wait(2)
        
        # Emphasize the need for error redistribution
        error_insight = Text(
            "GPTQ: Redistribute these errors to less important weights", 
            font_size=28,
            color=YELLOW
        )
        error_insight.next_to(error_viz, DOWN, buff=0.5)
        
        self.play(Write(error_insight))
        self.wait(2)
        self.play(
            FadeOut(original_weights), 
            FadeOut(quantized_weights), 
            FadeOut(error_viz),
            FadeOut(error_insight)
        )
    
    def create_weight_visualization(self, values, title, value_color=WHITE):
        n = len(values)
        group = VGroup()
        
        # Title
        title_text = Text(title, font_size=28)
        group.add(title_text)
        
        # Create squares for values
        squares = VGroup()
        for i, val in enumerate(values):
            square = Square(side_length=0.5)
            square.set_fill(color=BLUE_E if val >= 0 else RED_E, opacity=min(abs(val) + 0.1, 0.9))
            
            # Value text
            value_text = Text(f"{val:.3f}", font_size=12, color=value_color)
            value_text.move_to(square)
            
            # Group them
            cell = VGroup(square, value_text)
            if i > 0:
                cell.next_to(squares[-1], RIGHT, buff=0.1)
            squares.add(cell)
        
        squares.arrange(RIGHT, buff=0.1)
        squares.next_to(title_text, DOWN, buff=0.3)
        group.add(squares)
        
        # Add background
        background = SurroundingRectangle(group, color=GREY_D, fill_opacity=0.1, buff=0.2)
        full_group = VGroup(background, group)
        
        return full_group
        
    def visualize_error_redistribution(self):
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )

        # Visualize the Hessian and error redistribution
        hessian_title = Text("The Hessian Matrix and Error Redistribution using \n"
                             " Cholesky Decomposition", font_size=36)
        hessian_title.to_edge(UP)
        
        # Create Hessian matrix visualization
        hessian_values = np.array([
            [1.0, 0.3, 0.1, 0.05],
            [0.3, 0.9, 0.2, 0.1],
            [0.1, 0.2, 1.1, 0.25],
            [0.05, 0.1, 0.25, 0.95]
        ])
        
        hessian_matrix = self.create_matrix_visualization(
            hessian_values, 
            title="Hessian Matrix (H)",
            description="Captures weight importance\n & interactions"
        )
        
        hessian_inv = np.linalg.inv(hessian_values)
        hessian_inv_viz = self.create_matrix_visualization(
            hessian_inv, 
            title="Hessian Inverse (H⁻¹)",
            description="Used for error redistribution"
        )
        
        # Create Cholesky decomposition visualization
        chol = np.linalg.cholesky(hessian_inv).T
        chol_viz = self.create_matrix_visualization(
            chol, 
            title="Cholesky(H⁻¹)ᵀ",
            description="Efficient computation form"
        )
        
        # Arrange visualizations
        hessian_matrix.to_edge(LEFT).shift(UP)
        hessian_inv_viz.next_to(hessian_matrix, RIGHT, buff=0.3)
        chol_viz.next_to(hessian_inv_viz, RIGHT, buff=0.3)
        
        self.play(Write(hessian_title))
        self.play(Write(hessian_matrix))
        self.wait(1)
        self.play(Write(hessian_inv_viz))
        self.wait(1)
        # self.play(Write(chol_viz))
        # self.wait(2)

        # Add explanation about Cholesky decomposition
        cholesky_title = Text("Cholesky Decomposition", font_size=32, color=YELLOW)
        # cholesky_title.next_to(hessian_title, DOWN, buff=3.5)
        cholesky_title.next_to(hessian_inv_viz, RIGHT, buff=0.5).shift([0, 1.5, 0])
        
        cholesky_def = VGroup(
            MathTex(r"A = LL^*", font_size=32),
            Text(
                "L is a lower triangular matrix \n"
                "with positive diagonal entries", font_size=24)
        ).arrange(DOWN, buff=0.3)
        cholesky_def.next_to(cholesky_title, DOWN, buff=0.3)
        
        cholesky_benefits = VGroup(
            Text("Benefits:", font_size=26, color=BLUE),
            BulletedList(
                "Approximately twice as efficient as LU decomposition",
                "Enables fast numerical solutions",
                "Provides numerical stability for GPTQ algorithm",
                font_size=22
            )
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        cholesky_benefits.next_to(cholesky_def, DOWN, buff=0.4)
        
        self.play(Write(cholesky_title))
        self.play(Write(cholesky_def))
        self.play(Write(cholesky_benefits))
        self.wait(2)
        
        self.play(
            FadeOut(cholesky_title),
            FadeOut(cholesky_def),
            FadeOut(cholesky_benefits)
        )
        self.play(Write(chol_viz))
        self.wait(2)

        self.play(
            # FadeOut(hessian_title),
            FadeOut(hessian_matrix),
            FadeOut(hessian_inv_viz),
            FadeOut(chol_viz),
        )
        
        # Show error redistribution with animation
        error_vector = np.array([[0.1], [0.0], [0.0], [0.0]])
        error_viz = self.create_matrix_visualization(
            error_vector,
            title="Error Vector (e)",
            description="Quantization error"
        )
        # error_viz.next_to(hessian_matrix, DOWN, buff=0.1)
        error_viz.to_edge(LEFT).shift(UP)
        
        # Show redistribution calculation
        redistribution = chol @ error_vector
        redistribution_viz = self.create_matrix_visualization(
            redistribution,
            title="Redistributed Error",
            description="Cholesky(H⁻¹)ᵀ · e"
        )
        redistribution_viz.next_to(error_viz, RIGHT, buff=1)
        
        self.play(Write(error_viz))
        self.wait(1)
        
        redistribution_arrow = Arrow(
            error_viz.get_right(), 
            redistribution_viz.get_left(),
            buff=0.2
        )
        chol_label = Text("Apply", font_size=24).next_to(redistribution_arrow, UP, buff=0.1)
        
        self.play(
            Write(redistribution_arrow),
            Write(chol_label)
        )
        self.play(Write(redistribution_viz))
        self.wait(2)
        
        self.play(
            FadeOut(hessian_title),
            # FadeOut(hessian_matrix),
            # FadeOut(hessian_inv_viz),
            # FadeOut(chol_viz),
            FadeOut(error_viz),
            FadeOut(redistribution_arrow),
            FadeOut(chol_label),
            FadeOut(redistribution_viz)
        )
    
    def create_matrix_visualization(self, matrix, title, description=None):
        group = VGroup()
        
        # Title
        title_text = Text(title, font_size=28)
        group.add(title_text)
        
        # Add description if provided
        if description:
            desc_text = Text(description, font_size=16, color=GREY_A)
            desc_text.next_to(title_text, DOWN, buff=0.2)
            group.add(desc_text)
            matrix_anchor = desc_text
        else:
            matrix_anchor = title_text
        
        # Create matrix
        rows, cols = matrix.shape
        cells = VGroup()
        
        for i in range(rows):
            row_cells = VGroup()
            for j in range(cols):
                val = matrix[i, j]
                square = Square(side_length=0.3)
                square.set_fill(color=BLUE_E if val >= 0 else RED_E, opacity=min(abs(val) + 0.1, 0.9))
                
                # Value text
                value_text = Text(f"{val:.2f}", font_size=12)
                value_text.move_to(square)
                
                # Group them
                cell = VGroup(square, value_text)
                if j > 0:
                    cell.next_to(row_cells[-1], RIGHT, buff=0.05)
                row_cells.add(cell)
                
            if i > 0:
                row_cells.next_to(cells[-1], DOWN, buff=0.05)
            cells.add(row_cells)
        
        # Add matrix to group
        cells.next_to(matrix_anchor, DOWN, buff=0.4)
        group.add(cells)
        
        # Add background
        background = SurroundingRectangle(group, color=GREY_D, fill_opacity=0.1, buff=0.2)
        full_group = VGroup(background, group)
        
        return full_group
        
    def show_algorithm_steps(self):
        # Show the algorithm steps with pseudo-code
        algo_title = Text("GPTQ Algorithm Steps", font_size=36)
        algo_title.to_edge(UP)
        
        # Create algorithm steps
        algo_steps = self.create_algorithm_steps()
        algo_steps.next_to(algo_title, DOWN, buff=0.5)
        
        self.play(Write(algo_title))
        self.play(Write(algo_steps))
        self.wait(3)
        
        # Create animation showing block processing
        block_processing = self.create_block_processing_animation()
        block_processing.next_to(algo_steps, DOWN, buff=1)
        
        self.play(Write(block_processing))
        self.wait(2)
        
        self.play(
            FadeOut(algo_title),
            FadeOut(algo_steps),
            FadeOut(block_processing)
        )
    
    def create_algorithm_steps(self):
        pseudo_code = VGroup(
            Tex("1. $Q = 0$ (Initialize quantized weights)", font_size=24),
            Tex("2. $E = 0$ (Initialize block quantization errors)", font_size=24),
            Tex("3. $H^{-1} = Cholesky(H^{-1})^{T}$ (Precompute Hessian)", font_size=24),
            Tex("4. For each block i of columns:", font_size=24),
            Tex(" a. For each column j in block:", font_size=24),
            Tex("  i. $Q_{:,j} = quant(W_{:, j})$ (Quantize column)", font_size=24),
            Tex("  ii. $E_{:,j-i} = (W_{:,j} - Q_{:,j})$ / $[H^{-1}]_{jj}$ (compute the Error)", font_size=24),
            Tex("  iii. $W_{:,j:(i+B)} = W_{:,j:(i+B)} - E_{:,j-i} \cdot H^{-1}_{j,j:(i+B)}$ (update weights in block)", font_size=24),
            Tex(" b. Update all remaining weights", font_size=24)
        )
        
        # Arrange items vertically with alignment
        pseudo_code.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Add background rectangle
        background = SurroundingRectangle(pseudo_code, color=BLUE_D, fill_opacity=0.1, buff=0.3)
        return VGroup(background, pseudo_code)
    
    def create_block_processing_animation(self):
        # Create a visualization showing how blocks are processed
        n_cols = 8
        block_size = 2
        
        # Create weight matrix representation
        matrix_viz = VGroup()
        for i in range(n_cols):
            square = Square(side_length=0.5)
            square.set_stroke(WHITE, 2)
            square.set_fill(BLUE_E, opacity=0.3)
            
            # Position the squares horizontally
            if i > 0:
                square.next_to(matrix_viz[-1], RIGHT, buff=0.1)
                
            matrix_viz.add(square)
        
        # Label for the matrix
        matrix_label = Text("Weight Matrix Columns", font_size=24)
        matrix_label.next_to(matrix_viz, UP, buff=0.3)
        
        # Create arrows showing processing order
        block_labels = VGroup()
        arrows = VGroup()
        
        for block_idx in range(0, n_cols, block_size):
            # Highlight block
            block = VGroup(*[matrix_viz[i] for i in range(block_idx, min(block_idx + block_size, n_cols))])
            
            # Block label
            block_label = Text(f"Block {block_idx//block_size + 1}", font_size=20)
            block_label.next_to(block, DOWN, buff=0.3)
            block_labels.add(block_label)
            
            # Processing arrow
            if block_idx < n_cols - block_size:
                arrow = Arrow(
                    block.get_bottom() + DOWN * 0.5,
                    matrix_viz[min(n_cols-1, block_idx + block_size)].get_bottom() + DOWN * 0.5,
                    buff=0.2,
                    color=YELLOW
                )
                arrows.add(arrow)
        
        explanation = Text(
            "Process blocks left-to-right, redistributing errors to remaining weights", 
            font_size=20
        )
        explanation.next_to(block_labels, DOWN, buff=0.5)
        
        return VGroup(matrix_label, matrix_viz, block_labels, arrows, explanation)
        
    def conclusion(self):
        # Summarize the key benefits of GPTQ
        conclusion_title = Text("Benefits of GPTQ Algorithm", font_size=36)
        conclusion_title.to_edge(UP)
        
        benefits = VGroup(
            BulletedList(
                "Significantly reduces model size (4-bit quantization)",
                "Preserves model accuracy by intelligently redistributing errors",
                "Processes weights by importance using Hessian information",
                "Enables efficient inference on resource-constrained devices",
                "Minimal calibration data needed (few hundred examples)",
                font_size=28
            )
        )
        benefits.next_to(conclusion_title, DOWN, buff=0.5)
        
        final_equation = MathTex(
            r"\text{GPTQ: } \underset{\hat{W}}{\text{argmin}} \, \|WX - \hat{W}X\|^2_2",
            r"\text{ with error redistribution}",
            font_size=36
        )
        final_equation.next_to(benefits, DOWN, buff=0.8)
        
        self.play(Write(conclusion_title))
        self.play(Write(benefits))
        self.play(Write(final_equation))
        self.wait(3)
        
        # Final title
        final_title = Text("GPTQ: Efficient Post-Training Quantization", font_size=48)
        final_title.move_to(ORIGIN)
        
        self.play(
            FadeOut(conclusion_title),
            FadeOut(benefits),
            FadeOut(final_equation)
        )
        self.play(Write(final_title))
        self.wait(2)