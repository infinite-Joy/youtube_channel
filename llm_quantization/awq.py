"""
Usage

 python -m manim -pql .\awq.py AWQVisualization 
"""
from manim import *
import numpy as np

class AWQVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Activation-Aware Weight Quantization (AWQ)", font_size=40)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.6).to_edge(UP))
        
        # Introduction to LLM quantization challenge
        llm_structure = self.create_llm_architecture()
        self.play(Create(llm_structure))
        self.wait(1)
        self.play(
            FadeOut(title),
            FadeOut(llm_structure)
        )
        
        # Show problem with traditional weight-only quantization
        traditional_quantization = self.show_traditional_quantization()
        self.play(FadeIn(traditional_quantization))
        self.wait(1.5)
        self.play(FadeOut(traditional_quantization))
        
        # Explain AWQ concept
        awq_concept = self.explain_awq_concept()
        self.play(FadeIn(awq_concept))
        self.wait(2)
        
        # Create visualization of scaling weights and activations
        scaling_visualization = self.show_scaling_visualization()
        self.play(FadeOut(awq_concept), FadeIn(scaling_visualization))
        self.wait(2)
        
        # Show error reduction
        error_comparison = self.show_error_comparison()
        self.play(FadeOut(scaling_visualization), FadeIn(error_comparison))
        self.wait(2)
        
        # Explain channel sensitivity and optimization
        sensitivity_viz = self.show_channel_sensitivity()
        self.play(FadeOut(error_comparison), FadeIn(sensitivity_viz))
        self.wait(2)
        
        # Show AWQ integration with kernel fusion
        kernel_fusion = self.show_kernel_fusion()
        self.play(FadeOut(sensitivity_viz), FadeIn(kernel_fusion))
        self.wait(2)
        
        # Conclusion
        conclusion = self.show_conclusion()
        self.play(FadeOut(kernel_fusion), FadeIn(conclusion))
        self.wait(2)
        
        # Final title
        final_text = Text("AWQ: Better Accuracy at the Same Speed", font_size=36)
        self.play(FadeOut(conclusion), Write(final_text))
        self.wait(1)
    
    def create_llm_architecture(self):
        # Create a simplified diagram of an LLM architecture
        group = VGroup()
        
        # Boxes representing layers
        input_box = Rectangle(height=1, width=2, color=BLUE).shift(LEFT*3)
        hidden_box1 = Rectangle(height=1, width=2, color=GREEN).shift(LEFT*1)
        hidden_box2 = Rectangle(height=1, width=2, color=GREEN).shift(RIGHT*1)
        output_box = Rectangle(height=1, width=2, color=RED).shift(RIGHT*3)
        
        # Labels
        input_label = Text("Input", font_size=20).next_to(input_box, DOWN)
        hidden_label1 = Text("Hidden", font_size=20).next_to(hidden_box1, DOWN)
        hidden_label2 = Text("Hidden", font_size=20).next_to(hidden_box2, DOWN)
        output_label = Text("Output", font_size=20).next_to(output_box, DOWN)
        
        # Arrows
        arrow1 = Arrow(input_box.get_right(), hidden_box1.get_left())
        arrow2 = Arrow(hidden_box1.get_right(), hidden_box2.get_left())
        arrow3 = Arrow(hidden_box2.get_right(), output_box.get_left())
        
        # Large memory indicator
        memory_box = SurroundingRectangle(VGroup(hidden_box1, hidden_box2), color=YELLOW, buff=0.3)
        memory_text = Text("Large Memory Footprint", font_size=24, color=YELLOW).next_to(memory_box, UP)
        
        group.add(input_box, hidden_box1, hidden_box2, output_box, 
                 input_label, hidden_label1, hidden_label2, output_label,
                 arrow1, arrow2, arrow3, memory_box, memory_text)
        
        return group
    
    def show_traditional_quantization(self):
        group = VGroup()
        
        title = Text("Traditional Weight-Only Quantization", font_size=30)
        # title.to_edge(UP).shift(DOWN*0.5)
        title.to_edge(UP)
        
        # Create a weight matrix visualization
        matrix = self.create_matrix_visualization(4, 4, color_range=[BLUE_D, BLUE_A], title="Weight Matrix (FP16)")
        # matrix.scale(1.5).shift(LEFT*1 + DOWN*0.3)
        matrix.scale(1.5).to_edge(LEFT)
        
        # Create a quantized matrix
        quantized = self.create_matrix_visualization(4, 4, color_range=[GREEN_D, GREEN_A], title="Quantized (INT4)", 
                                                    noise_factor=0.4)
        # quantized.scale(1.5).shift(RIGHT*1 + DOWN*0.3)
        quantized.scale(1.5).to_edge(RIGHT)
        
        # Arrow showing transformation
        arrow = Arrow(matrix.get_right() + RIGHT*0.2, quantized.get_left() + LEFT*0.5, buff=0.2)
        quant_text = Text("Quantize", font_size=24).next_to(arrow, UP, buff=0.1)
        
        # Error visualization
        error_text = Text("Quantization Error", font_size=26, color=RED).shift(DOWN*2.5)
        error_eq = MathTex(r"\text{Error} \propto \Delta \cdot |x|", font_size=30).next_to(error_text, DOWN)
        
        group.add(title, matrix, quantized, arrow, quant_text, error_text, error_eq)
        return group
    
    def explain_awq_concept(self):
        group = VGroup()
        
        title = Text("AWQ Mathematical Foundation", font_size=30)
        title.to_edge(UP).shift(DOWN*0.5)
        
        # Explain quantization process first
        quant_title = Text("Symmetric Quantization", font_size=24)
        quant_title.shift(UP*1.5)
        
        quant_formula = MathTex(
            r"Q(w) = \Delta \cdot \text{Round}\left(\frac{w}{\Delta}\right)",
            font_size=28
        )
        quant_formula.next_to(quant_title, DOWN, buff=0.3)
        
        delta_def = MathTex(
            r"\text{where} \quad \Delta = \frac{\max(|w|)}{2^{b-1} - 1}",
            font_size=28
        )
        delta_def.next_to(quant_formula, DOWN, buff=0.2)
        delta_exp = Text("Δ = quantization scale, b = bit width", font_size=20)
        delta_exp.next_to(delta_def, DOWN, buff=0.1)
        
        # Add error formula and insight
        error_title = Text("Quantization Error Analysis", font_size=24)
        error_title.next_to(delta_exp, DOWN, buff=0.4)
        
        error_formula = MathTex(
            r"\text{Error}(Q(w)x) = \Delta \cdot \text{RoundErr}\left(\frac{w}{\Delta}\right) \cdot x",
            font_size=28
        )
        error_formula.next_to(error_title, DOWN, buff=0.3)
        
        round_err = MathTex(
            r"\text{RoundErr}\left(\frac{w}{\Delta}\right) = \frac{w}{\Delta} - \text{Round}\left(\frac{w}{\Delta}\right)",
            font_size=26
        )
        round_err.next_to(error_formula, DOWN, buff=0.2)
        
        # Highlight the key insight
        insight_box = SurroundingRectangle(VGroup(error_formula, round_err), corner_radius=0.2, color=YELLOW)
        
        # Key insight
        insight_text = Text("Key Insight: Error is proportional to activation magnitude |x|", 
                           font_size=26, color=RED)
        insight_text.next_to(insight_box, DOWN, buff=0.3)
        
        solution = Text("Solution: Scale weights up, scale activations down", font_size=26, color=GREEN)
        solution.next_to(insight_text, DOWN, buff=0.3)
        
        group.add(title, quant_title, quant_formula, delta_def, delta_exp, 
                 error_title, error_formula, round_err, insight_box, insight_text, solution)
        return group
    
    def show_scaling_visualization(self):
        group = VGroup()
        
        title = Text("AWQ: Mathematical Transformation", font_size=30)
        title.to_edge(UP)
        
        # Original computation
        original_eq = MathTex(r"y = wx", font_size=30)
        original_eq.shift(UP*1)
        
        # Mathematical identity
        identity_eq = MathTex(r"y = w \cdot \text{diag}(s) \cdot \text{diag}(s)^{-1} \cdot x", font_size=30)
        identity_eq.next_to(original_eq, DOWN, buff=0.5)
        
        identity_box = SurroundingRectangle(identity_eq, corner_radius=0.2, color=BLUE)
        identity_text = Text("Mathematical Identity (Exact Equivalence)", font_size=20, color=BLUE)
        identity_text.next_to(identity_box, RIGHT, buff=0.3)
        
        # Quantized version
        quant_eq = MathTex(r"y \approx Q(w \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} \cdot x", font_size=30)
        quant_eq.next_to(identity_eq, DOWN, buff=0.5)
        
        # Expand step by step
        expand_title = Text("Error Analysis", font_size=24)
        expand_title.next_to(quant_eq, DOWN, buff=0.5)
        
        # Original quantization error
        original_error = MathTex(
            r"\text{Std Error} = \Delta \cdot \text{RoundErr} \cdot |x|",
            font_size=26
        )
        original_error.next_to(expand_title, DOWN, buff=0.3)
        
        # AWQ quantization error
        awq_error = MathTex(
            r"\text{AWQ Error} = \Delta' \cdot \text{RoundErr} \cdot \frac{|x|}{s}",
            font_size=26
        )
        awq_error.next_to(original_error, DOWN, buff=0.3)
        
        # Error reduction formula
        reduction_formula = MathTex(
            r"\text{Error Reduction} = \frac{\Delta'}{\Delta} \cdot \frac{1}{s}",
            font_size=26
        )
        reduction_formula.next_to(awq_error, DOWN, buff=0.3)
        
        # Add condition for error reduction
        condition = MathTex(
            r"\text{Error is reduced when } \frac{\Delta'}{\Delta} < s",
            font_size=26, color=GREEN
        )
        condition.next_to(reduction_formula, DOWN, buff=0.3)
        
        group.add(title, original_eq, identity_eq, identity_box, identity_text, 
                 quant_eq, expand_title, original_error, awq_error, 
                 reduction_formula, condition)
        return group
    
    def show_error_comparison(self):
        group = VGroup()
        
        title = Text("Quantization Scale Analysis", font_size=30)
        title.to_edge(UP)
        
        # Quantization scales
        std_scale = MathTex(
            r"\Delta = \frac{\max(|w|)}{2^{b-1} - 1}",
            font_size=28
        )
        std_scale.shift(UP*1.5)
        
        awq_scale = MathTex(
            r"\Delta' = \frac{\max(|w \cdot \text{diag}(s)|)}{2^{b-1} - 1}",
            font_size=28
        )
        awq_scale.next_to(std_scale, DOWN, buff=0.5)
        
        # How scale factor affects quantization
        scale_analysis = VGroup(
            Text("When scaling weights uniformly by s:", font_size=24),
            MathTex(r"\Delta' = s \cdot \Delta", font_size=26),
            Text("Error reduction factor:", font_size=24),
            MathTex(r"\frac{\Delta'}{\Delta} \cdot \frac{1}{s} = \frac{s \cdot \Delta}{\Delta} \cdot \frac{1}{s} = 1", font_size=26),
            Text("No improvement with uniform scaling", font_size=24, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        scale_analysis.next_to(awq_scale, DOWN, buff=0.6)
        
        # Channel-wise scaling advantage
        advantage_title = Text("Channel-wise Scaling Advantage:", font_size=24, color=GREEN)
        advantage_title.next_to(scale_analysis, DOWN, buff=0.4)
        
        advantage_points = VGroup(
            MathTex(r"1.\text{ Apply larger } s_i \text{ to channels with larger } |x_i|", font_size=24),
            MathTex(r"2.\text{ }\max(|w \cdot \text{diag}(s)|) \text{ dominated by some channels}", font_size=24),
            MathTex(r"3.\text{ Other channels can use lower } \frac{\Delta'}{s_i} \text{ for those activations}", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        advantage_points.next_to(advantage_title, DOWN, buff=0.3)
        
        group.add(title, std_scale, awq_scale, scale_analysis, advantage_title, advantage_points)
        return group
    
    def show_channel_sensitivity(self):
        group = VGroup()
        
        title = Text("Optimal Channel-wise Scaling Factor Derivation", font_size=30)
        title.to_edge(UP)
        
        # Optimization objective
        objective_title = Text("Optimization Objective:", font_size=24)
        objective_title.shift(UP*1.5)
        
        objective = MathTex(
            r"s^* = \arg\min_s L(s)",
            font_size=28
        )
        objective.next_to(objective_title, DOWN, buff=0.2)
        
        loss_fn = MathTex(
            r"L(s) = \mathbb{E}_{x \sim X} \left[ \left\| Q(w \cdot \text{diag}(s))(\text{diag}(s)^{-1} \cdot x) - wx \right\|^2 \right]",
            font_size=28
        )
        loss_fn.next_to(objective, DOWN, buff=0.3)
        
        # Explanation about gradient challenges
        grad_explain = Text("Challenge: Non-differentiable quantization function Q(·)", font_size=22, color=RED)
        grad_explain.next_to(loss_fn, DOWN, buff=0.4)
        
        # Practical solution using heuristics
        practical_title = Text("Practical Solution - Heuristic Approach:", font_size=24)
        practical_title.next_to(grad_explain, DOWN, buff=0.4)
        
        practical_eq = MathTex(
            r"s^* = s_x^{\alpha^*} \quad \text{where} \quad \alpha^* = \arg\min_\alpha L(s_x^\alpha)",
            font_size=28
        )
        practical_eq.next_to(practical_title, DOWN, buff=0.2)
        
        s_x_def = MathTex(
            r"s_x = \mathbb{E}_{x \sim X}[|x|] \quad \text{(average activation magnitude)}",
            font_size=24
        )
        s_x_def.next_to(practical_eq, DOWN, buff=0.2)
        
        alpha_search = Text("α* found via grid search in range [0,1]", font_size=22)
        alpha_search.next_to(s_x_def, DOWN, buff=0.2)
        
        # Key insight about sensitivity
        insight = Text("Key Insight: Apply larger scaling to channels with larger input activations", 
                      font_size=24, color=GREEN)
        insight.next_to(alpha_search, DOWN, buff=0.4)
        
        group.add(title, objective_title, objective, loss_fn, grad_explain, 
                 practical_title, practical_eq, s_x_def, alpha_search, insight)
        return group
    
    def show_kernel_fusion(self):
        group = VGroup()
        
        title = Text("Mathematical Implementation with Kernel Fusion", font_size=30)
        title.to_edge(UP)
        
        # Describe the implementation challenge
        challenge_text = Text("Challenge: How to implement activation scaling efficiently?", font_size=24)
        challenge_text.shift(UP*1.5)
        
        # Original equations
        eqs_title = Text("Original operations:", font_size=22)
        eqs_title.next_to(challenge_text, DOWN, buff=0.4)
        
        layer_norm_eq = MathTex(
            r"\hat{x} = \gamma \cdot \frac{x - \mu}{\sigma} + \beta",
            font_size=26
        )
        layer_norm_eq.next_to(eqs_title, DOWN, buff=0.2)
        
        scaling_eq = MathTex(
            r"x' = \text{diag}(s)^{-1} \cdot \hat{x}",
            font_size=26
        )
        scaling_eq.next_to(layer_norm_eq, DOWN, buff=0.3)
        
        matmul_eq = MathTex(
            r"y = Q(w \cdot \text{diag}(s)) \cdot x'",
            font_size=26
        )
        matmul_eq.next_to(scaling_eq, DOWN, buff=0.3)
        
        # Fusion optimization
        fusion_title = Text("Fused implementation:", font_size=22)
        fusion_title.next_to(matmul_eq, DOWN, buff=0.4)
        
        fused_eq = MathTex(
            r"\hat{x} = \left(\gamma \cdot \text{diag}(s)^{-1}\right) \cdot \frac{x - \mu}{\sigma} + \left(\beta \cdot \text{diag}(s)^{-1}\right)",
            font_size=24
        )
        fused_eq.next_to(fusion_title, DOWN, buff=0.2)
        
        # Explanation of efficiency
        efficiency_text = Text("Combine LayerNorm parameters with scaling factors:", font_size=22)
        efficiency_text.next_to(fused_eq, DOWN, buff=0.3)
        
        gamma_prime = MathTex(
            r"\gamma' = \gamma \cdot \text{diag}(s)^{-1}",
            font_size=24
        )
        gamma_prime.next_to(efficiency_text, DOWN, buff=0.1)
        
        beta_prime = MathTex(
            r"\beta' = \beta \cdot \text{diag}(s)^{-1}",
            font_size=24
        )
        beta_prime.next_to(gamma_prime, DOWN, buff=0.1)
        
        conclusion = Text("No additional runtime overhead compared to standard quantization", 
                          font_size=24, color=GREEN)
        conclusion.next_to(beta_prime, DOWN, buff=0.2)
        
        group.add(title, challenge_text, eqs_title, layer_norm_eq, scaling_eq, matmul_eq,
                 fusion_title, fused_eq, efficiency_text, gamma_prime, beta_prime, conclusion)
        return group
    
    def show_conclusion(self):
        group = VGroup()
        
        title = Text("AWQ Benefits", font_size=30)
        title.to_edge(UP).shift(DOWN*0.5)
        
        # Benefits bullet points
        benefits = VGroup(
            Text("• Same memory efficiency as regular W4A16", font_size=24),
            Text("• Same computational speed", font_size=24),
            Text("• Better accuracy through activation-aware scaling", font_size=24),
            Text("• Easy integration with existing quantization pipelines", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        benefits.shift(DOWN*0.5)
        
        group.add(title, benefits)
        return group
    
    # Helper methods
    def create_matrix_visualization(self, rows, cols, color_range=[BLUE_D, BLUE_A], title=None, noise_factor=0):
        matrix = VGroup()
        
        cells = VGroup()
        for i in range(rows):
            for j in range(cols):
                val = np.random.random() + noise_factor * np.random.normal(0, 0.3)
                val = max(0, min(1, val))  # Clamp between 0 and 1
                
                color = interpolate_color(color_range[0], color_range[1], val)
                cell = Square(side_length=0.5, fill_opacity=0.8, fill_color=color, stroke_width=1, stroke_color=WHITE)
                cell.move_to([j*0.5, -i*0.5, 0])
                cells.add(cell)
        
        matrix.add(cells)
        
        if title:
            title_text = Text(title, font_size=20)
            title_text.next_to(cells, UP, buff=0.2)
            matrix.add(title_text)
        
        return matrix
    
    def create_weight_activation_scaling(self):
        group = VGroup()
        
        # Create weight matrix and activation vector
        weight_matrix = self.create_matrix_visualization(4, 4, color_range=[BLUE_D, BLUE_A], title="W")
        weight_matrix.scale(0.6).shift(LEFT*3.5)
        
        activation_vector = VGroup(*[
            Square(side_length=0.5, fill_opacity=0.8, 
                  fill_color=interpolate_color(RED_D, RED_A, np.random.random()),
                  stroke_width=1, stroke_color=WHITE)
            for _ in range(4)
        ]).arrange(DOWN, buff=0).scale(0.6)
        
        activation_vector.next_to(weight_matrix, RIGHT)
        act_label = Text("x", font_size=16).next_to(activation_vector, UP, buff=0.1)
        
        # Scaling operations
        scale_up_arrow = Arrow(weight_matrix.get_top() + UP*0.1, weight_matrix.get_top() + UP*0.5)
        scale_up_text = Text("Scale up (s)", font_size=16, color=GREEN).next_to(scale_up_arrow, UP, buff=0.1)
        
        scale_down_arrow = Arrow(activation_vector.get_top() + UP*0.1, activation_vector.get_top() + UP*0.5)
        scale_down_text = Text("Scale down (1/s)", font_size=16, color=YELLOW).next_to(scale_down_arrow, UP, buff=0.1)
        
        # Scaled versions
        weight_matrix_scaled = self.create_matrix_visualization(4, 4, color_range=[GREEN_D, GREEN_A], title="W·diag(s)")
        weight_matrix_scaled.scale(0.6).shift(RIGHT*0.5)
        
        activation_vector_scaled = VGroup(*[
            Square(side_length=0.5, fill_opacity=0.6, 
                  fill_color=interpolate_color(YELLOW_D, YELLOW, np.random.random() * 0.6),  # Lighter to show "scaled down"
                  stroke_width=1, stroke_color=WHITE)
            for _ in range(4)
        ]).arrange(DOWN, buff=0).scale(0.6)
        
        activation_vector_scaled.next_to(weight_matrix_scaled, RIGHT)
        act_scaled_label = Text("diag(s)⁻¹·x", font_size=16).next_to(activation_vector_scaled, UP, buff=0.1)
        
        # Output equals
        equals = Text("=", font_size=30).shift(RIGHT*2.5)
        
        # Result
        result_vector = VGroup(*[
            Square(side_length=0.5, fill_opacity=0.8, 
                  fill_color=interpolate_color(PURPLE_D, PURPLE_A, np.random.random()),
                  stroke_width=1, stroke_color=WHITE)
            for _ in range(4)
        ]).arrange(DOWN, buff=0).scale(0.6)
        
        result_vector.next_to(equals, RIGHT)
        result_label = Text("y", font_size=16).next_to(result_vector, UP, buff=0.1)
        
        group.add(weight_matrix, activation_vector, act_label, 
                  scale_up_arrow, scale_up_text, scale_down_arrow, scale_down_text,
                  weight_matrix_scaled, activation_vector_scaled, act_scaled_label,
                  equals, result_vector, result_label)
        
        return group
    
    def create_error_graph(self):
        group = VGroup()
        
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": WHITE},
            x_length=5,
            y_length=3
        )
        
        # Add labels
        x_label = Text("Channel Index", font_size=20).next_to(axes, DOWN)
        y_label = Text("Error", font_size=20).next_to(axes, LEFT).rotate(90 * DEGREES)
        
        # Create plots for standard and AWQ errors
        std_points = [(x, 5 + np.random.normal(0, 0.5)) for x in range(1, 10)]
        awq_points = [(x, 2 + np.random.normal(0, 0.3)) for x in range(1, 10)]
        
        std_dots = VGroup(*[Dot(axes.coords_to_point(x, y), color=RED) for x, y in std_points])
        awq_dots = VGroup(*[Dot(axes.coords_to_point(x, y), color=GREEN) for x, y in awq_points])
        
        # Connect dots with lines
        std_line = VGroup(*[
            Line(
                axes.coords_to_point(std_points[i][0], std_points[i][1]),
                axes.coords_to_point(std_points[i+1][0], std_points[i+1][1]),
                color=RED
            )
            for i in range(len(std_points)-1)
        ])
        
        awq_line = VGroup(*[
            Line(
                axes.coords_to_point(awq_points[i][0], awq_points[i][1]),
                axes.coords_to_point(awq_points[i+1][0], awq_points[i+1][1]),
                color=GREEN
            )
            for i in range(len(awq_points)-1)
        ])
        
        # Add legend
        std_legend = Dot(color=RED)
        std_legend_text = Text("Standard Quantization", font_size=16, color=RED).next_to(std_legend, RIGHT)
        std_legend_group = VGroup(std_legend, std_legend_text).to_corner(UR)
        
        awq_legend = Dot(color=GREEN)
        awq_legend_text = Text("AWQ", font_size=16, color=GREEN).next_to(awq_legend, RIGHT)
        awq_legend_group = VGroup(awq_legend, awq_legend_text).next_to(std_legend_group, DOWN, aligned_edge=LEFT)
        
        # Add reduction label
        reduction_arrow = Arrow(
            axes.coords_to_point(5, 5),
            axes.coords_to_point(5, 2.5),
            color=YELLOW
        )
        reduction_text = Text("Reduced Error", font_size=16, color=YELLOW).next_to(reduction_arrow, RIGHT)
        
        group.add(axes, x_label, y_label, std_dots, awq_dots, std_line, awq_line, 
                  std_legend_group, awq_legend_group, reduction_arrow, reduction_text)
        
        return group
    
    def create_channel_sensitivity_viz(self):
        group = VGroup()
        
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": WHITE},
            x_length=6,
            y_length=3
        )
        
        # Add labels
        x_label = Text("Channel Index", font_size=20).next_to(axes, DOWN)
        y_label = Text("Sensitivity", font_size=20).next_to(axes, LEFT).rotate(90 * DEGREES)
        
        # Create sensitivity plot (some channels are more important)
        sensitivity_points = [(x, 2 + 8 * np.random.beta(2, 5)) for x in range(1, 10)]
        
        # Sort points by sensitivity for clearer visualization
        sensitivity_points.sort(key=lambda p: p[1], reverse=True)
        
        # Reassign x coordinates after sorting
        sensitivity_points = [(i+1, y) for i, (_, y) in enumerate(sensitivity_points)]
        
        sensitivity_dots = VGroup(*[
            Dot(axes.coords_to_point(x, y), 
                color=interpolate_color(BLUE, RED, y/10),
                radius=0.08)
            for x, y in sensitivity_points
        ])
        
        # Show scaling factors proportional to sensitivity
        scaling_factors = VGroup()
        for i, (x, y) in enumerate(sensitivity_points):
            factor = 1 + 4 * (y / 10)  # Scaling factor proportional to sensitivity
            text = Text(f"s={factor:.2f}", font_size=16)
            text.next_to(axes.coords_to_point(x, y), UP, buff=0.1)
            scaling_factors.add(text)
        
        # Add explanation
        explanation = Text("Channels with higher activation magnitude need larger scaling factors", 
                          font_size=20).next_to(axes, UP, buff=0.5)
        
        group.add(axes, x_label, y_label, sensitivity_dots, scaling_factors, explanation)
        
        return group
    
    def create_kernel_fusion_diagram(self):
        group = VGroup()
        
        # Create boxes for different operations
        layer_norm = Rectangle(height=1, width=2.5, color=BLUE)
        layer_norm_text = Text("Layer Norm", font_size=20).move_to(layer_norm)
        
        scaling = Rectangle(height=1, width=2.5, color=YELLOW)
        scaling_text = Text("Activation Scaling", font_size=20).move_to(scaling)
        
        matmul = Rectangle(height=1, width=2.5, color=GREEN)
        matmul_text = Text("Quantized MatMul", font_size=20).move_to(matmul)
        
        # Original pipeline
        original_title = Text("Original Pipeline", font_size=24)
        original_title.to_edge(LEFT).shift(UP*1.5 + RIGHT*2)
        
        layer_norm_orig = layer_norm.copy()
        layer_norm_text_orig = layer_norm_text.copy()
        scaling_orig = scaling.copy()
        scaling_text_orig = scaling_text.copy()
        matmul_orig = matmul.copy()
        matmul_text_orig = matmul_text.copy()
        
        orig_group = VGroup(layer_norm_orig, layer_norm_text_orig, scaling_orig, scaling_text_orig, 
                           matmul_orig, matmul_text_orig)
        orig_group.arrange(DOWN, buff=0.5)
        orig_group.next_to(original_title, DOWN, buff=0.5)
        
        # Arrows for original
        arrow1_orig = Arrow(layer_norm_orig.get_bottom(), scaling_orig.get_top())
        arrow2_orig = Arrow(scaling_orig.get_bottom(), matmul_orig.get_top())
        
        # Optimized pipeline with fusion
        optimized_title = Text("Optimized Pipeline (Fused)", font_size=24)
        optimized_title.to_edge(RIGHT).shift(UP*1.5 + LEFT*2)
        
        fused_op = Rectangle(height=1.5, width=3, color=PURPLE)
        fused_text = Text("Fused LayerNorm\n+ Activation Scaling", font_size=20).move_to(fused_op)
        
        matmul_opt = matmul.copy()
        matmul_text_opt = matmul_text.copy()
        
        opt_group = VGroup(fused_op, fused_text, matmul_opt, matmul_text_opt)
        opt_group.arrange(DOWN, buff=0.5)
        opt_group.next_to(optimized_title, DOWN, buff=0.5)
        
        # Arrow for optimized
        arrow_opt = Arrow(fused_op.get_bottom(), matmul_opt.get_top())
        
        # Performance benefit
        benefit_text = Text("Same computational cost as standard quantization", font_size=20, color=GREEN)
        benefit_text.shift(DOWN*2.5)
        
        group.add(original_title, orig_group, arrow1_orig, arrow2_orig,
                 optimized_title, opt_group, arrow_opt, benefit_text)
        
        return group