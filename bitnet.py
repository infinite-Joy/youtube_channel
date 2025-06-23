# To run this animation, save as bitnet_animation.py and run:
# manim -pql bitnet_animation.py BitNetQuantization

from manim import *
import numpy as np

class BitNetQuantizationAnimation(Scene):
    def construct(self):
        ## Scene 1: Introduction - The Quantization Journey
        #self.scene1_introduction()
        #self.wait(1)
        #self.clear()
        
        ## Scene 2: The Problem with 2-bit Quantization
        #self.scene2_problem_with_2bit()
        #self.wait(1)
        #self.clear()
        
        ## Scene 3: The Ternary Solution Discovery
        #self.scene3_ternary_solution()
        #self.wait(1)
        #self.clear()
        
        ## Scene 4: Matrix Multiplication Magic
        #self.scene4_matrix_multiplication()
        #self.wait(1)
        #self.clear()
        
        ## Scene 6: Weight Quantization Function
        #self.scene6_weight_quantization()
        #self.wait(1)
        #self.clear()
        
        ## Scene 7: Activation Quantization Function
        #self.scene7_activation_quantization()
        #self.wait(1)
        #self.clear()
        
        # Scene 8: BitLinear Architecture
        self.transformer_architecture_with_bitlinear()
        self.wait(1)
        self.clear()
        
    def scene1_introduction(self):
        """Scene 1: Introduction - The Quantization Journey"""
        title = Text("The Quantization Journey", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create timeline
        timeline = Line(LEFT * 5, RIGHT * 5, color=WHITE)
        timeline.shift(DOWN * 0.5)
        self.play(Create(timeline))
        
        # FP16 representation
        fp16_block = Rectangle(width=3, height=1.5, color=ORANGE, fill_opacity=0.7)
        fp16_block.shift(LEFT * 3 + DOWN * 0.5)
        fp16_label = Text("FP16\n16-bit", font_size=24, color=WHITE)
        fp16_label.move_to(fp16_block)
        fp16_memory = Text("32GB", font_size=16, color=ORANGE)
        fp16_memory.next_to(fp16_block, DOWN)
        
        # INT4 representation
        int4_block = Rectangle(width=2, height=1, color=GREEN, fill_opacity=0.7)
        int4_block.shift(DOWN * 0.5)
        int4_label = Text("INT4\n4-bit", font_size=20, color=WHITE)
        int4_label.move_to(int4_block)
        int4_memory = Text("8GB", font_size=16, color=GREEN)
        int4_memory.next_to(int4_block, DOWN)
        
        # Ternary representation
        ternary_block = Rectangle(width=1, height=0.6, color=BLUE, fill_opacity=0.7)
        ternary_block.shift(RIGHT * 3 + DOWN * 0.5)
        ternary_label = Text("Ternary\n1.58-bit", font_size=16, color=WHITE)
        ternary_label.move_to(ternary_block)
        ternary_memory = Text("1.6GB", font_size=16, color=BLUE)
        ternary_memory.next_to(ternary_block, DOWN)
        
        # Animate progression
        self.play(
            FadeIn(fp16_block), Write(fp16_label), Write(fp16_memory)
        )
        self.wait(0.5)
        
        self.play(
            Transform(fp16_block.copy(), int4_block),
            Write(int4_label), Write(int4_memory)
        )
        self.wait(0.5)
        
        self.play(
            Transform(int4_block.copy(), ternary_block),
            Write(ternary_label), Write(ternary_memory)
        )
        
        # Show ternary values
        ternary_values = Text("{-1, 0, 1}", font_size=36, color=YELLOW)
        ternary_values.shift(DOWN * 2.5)
        self.play(Write(ternary_values))
        self.wait(2)

    def scene2_problem_with_2bit(self):
        """Scene 2: The Problem with 2-bit Quantization"""
        title = Text("The Problem with 2-bit Quantization", font_size=40, color=RED)
        title.to_edge(UP)
        self.play(Write(title))
        
        # 1-bit system attempt
        onebit_values = Text("1-bit: {-1, 1}", font_size=32, color=WHITE)
        # onebit_values.shift(UP * 1)
        onebit_values.next_to(title, DOWN, buff=0.5)
        self.play(Write(onebit_values))
        
        # Neural network degradation
        # Create simple network representation
        # nodes_input = [Circle(radius=0.3, color=BLUE, fill_opacity=0.7).shift(LEFT * 3 + UP * i) for i in [-1, 0, 1]]
        # nodes_hidden = [Circle(radius=0.3, color=GREEN, fill_opacity=0.7).shift(UP * i * 0.7) for i in [-1, 0, 1]]
        # nodes_output = [Circle(radius=0.3, color=YELLOW, fill_opacity=0.7).shift(RIGHT * 3 + UP * i * 0.7) for i in [-0.5, 0.5]]
        nodes_input = [Circle(radius=0.25, color=BLUE, fill_opacity=0.7).shift(LEFT * 2 + DOWN * i * 0.7) for i in [-1, 0, 1]]
        nodes_hidden = [Circle(radius=0.25, color=GREEN, fill_opacity=0.7).shift(DOWN * i * 0.7) for i in [-1, 0, 1]]
        nodes_output = [Circle(radius=0.25, color=YELLOW, fill_opacity=0.7).shift(RIGHT * 2 + DOWN * i * 0.7) for i in [-0.5, 0.5]]

        # Create connections
        connections = []
        for i_node in nodes_input:
            for h_node in nodes_hidden:
                line = Line(i_node.get_center(), h_node.get_center(), color=WHITE, stroke_width=2)
                connections.append(line)
        
        for h_node in nodes_hidden:
            for o_node in nodes_output:
                line = Line(h_node.get_center(), o_node.get_center(), color=WHITE, stroke_width=2)
                connections.append(line)
        
        # Display network
        network_group = VGroup(*nodes_input, *nodes_hidden, *nodes_output, *connections)
        # network_group.shift(DOWN * 1)
        network_group.next_to(onebit_values, DOWN, buff=0.8)
        self.play(Create(network_group))
        
        # Show degradation
        self.wait(1)
        for i, connection in enumerate(connections):
            if i % 2 == 0:  # Randomly degrade some connections
                self.play(connection.animate.set_stroke(color=RED, opacity=0.3), run_time=0.1)
        
        # Performance graph
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 100, 20],
            x_length=4,
            y_length=2,
            axis_config={"color": WHITE}
        )
        # axes.shift(DOWN * 3.5 + RIGHT * 2)
        
        # Declining performance curve
        performance_curve = axes.plot(lambda x: 90 * np.exp(-0.3 * x), color=RED, x_range=[0, 8])
        
        axes_labels = axes.get_axis_labels(x_label="Training Steps", y_label="Accuracy %")
        graph_group = VGroup(axes, axes_labels, performance_curve)
        graph_group.scale(0.9)
        graph_group.next_to(network_group, DOWN, buff=0.8)
        
        self.play(Create(axes), Write(axes_labels))
        self.play(Create(performance_curve))
        
        # Red X mark
        red_x = Text("✗", font_size=60, color=RED)
        # red_x.shift(LEFT * 2 + DOWN * 2)
        red_x.next_to(graph_group, LEFT, buff=0.5)
        self.play(Write(red_x))
        self.wait(2)

    def scene3_ternary_solution(self):
        """Scene 3: The Ternary Solution Discovery & 1.58-Bit Magic"""
        title = Text("The Ternary Breakthrough", font_size=44, color=GOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show the evolution
        binary_text = Text("{-1, 1}", font_size=36, color=WHITE)
        binary_text.shift(LEFT * 2 + UP * 1)
        self.play(Write(binary_text))
        
        # Arrow
        arrow = Arrow(LEFT * 0.5 + UP * 1, RIGHT * 0.5 + UP * 1, color=YELLOW)
        self.play(Create(arrow))
        
        # Add zero
        ternary_text = Text("{-1, 0, 1}", font_size=36, color=GREEN)
        ternary_text.shift(RIGHT * 2 + UP * 1)
        self.play(Write(ternary_text))
        
        # Eureka moment
        eureka = Text("💡 EUREKA!", font_size=48, color=GOLD)
        eureka.shift(UP * 2)
        self.play(Write(eureka))
        
        # Information theory explanation
        info_title = Text("Why 1.58 bits?", font_size=32, color=BLUE)
        info_title.shift(DOWN * 0.5)
        self.play(Write(info_title))
        
        # Binary examples
        binary_examples = VGroup(
            Text("1-bit: 2¹ = 2 values → {0, 1}", font_size=24, color=WHITE),
            Text("2-bit: 2² = 4 values → {00, 01, 10, 11}", font_size=24, color=WHITE),
            Text("10-bit: 2¹⁰ = 1024 values", font_size=24, color=WHITE)
        )
        binary_examples.arrange(DOWN, buff=0.3)
        binary_examples.shift(DOWN * 2)
        
        for example in binary_examples:
            self.play(Write(example), run_time=0.8)
        
        # The challenge
        challenge = Text("Need 3 values: 2¹ = 2 (too few), 2² = 4 (wasteful)", font_size=24, color=YELLOW)
        challenge.shift(DOWN * 3.5)
        self.play(Write(challenge))
        
        # The solution
        solution = Text("log₂(3) ≈ 1.58 bits per ternary value", font_size=28, color=GREEN)
        solution.shift(DOWN * 4.2)
        self.play(Write(solution))
        self.wait(3)

    def scene4_matrix_multiplication(self):
        """Scene 4: Matrix Multiplication Magic"""
        title = Text("Matrix Multiplication Magic", font_size=40, color=PURPLE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Traditional multiplication (left side)
        trad_title = Text("Traditional", font_size=24, color=WHITE)
        trad_title.shift(LEFT * 4 + UP * 2)
        self.play(Write(trad_title))
        
        # Create traditional matrix with floating point values
        trad_matrix = MathTex(r"\begin{bmatrix} 0.7 & -0.3 \\ 0.1 & -0.9 \end{bmatrix}", color=WHITE)
        trad_matrix.shift(LEFT * 4 + UP * 0.5)
        
        input_vector = MathTex(r"\begin{bmatrix} 2.1 \\ 1.4 \end{bmatrix}", color=BLUE)
        input_vector.shift(LEFT * 2 + UP * 0.5)
        
        self.play(Write(trad_matrix), Write(input_vector))
        
        # Complex calculation
        complex_calc = Text("Many floating-point\nmultiplications", font_size=20, color=RED)
        complex_calc.shift(LEFT * 4 + DOWN * 0.5)
        self.play(Write(complex_calc))
        
        # BitNet approach (right side)
        bitnet_title = Text("BitNet", font_size=24, color=GREEN)
        bitnet_title.shift(RIGHT * 4 + UP * 2)
        self.play(Write(bitnet_title))
        
        # Ternary weight matrix with color coding
        weight_vals = [["1", "-1"], ["0", "1"]]
        weight_colors = [[GREEN, RED], [GRAY, GREEN]]
        
        bitnet_matrix = VGroup()
        for i in range(2):
            for j in range(2):
                val = Text(weight_vals[i][j], font_size=24, color=weight_colors[i][j])
                val.shift(RIGHT * 3.5 + j * 0.5 * RIGHT + UP * (1 - i * 0.5))
                bitnet_matrix.add(val)
        
        # Matrix brackets
        left_bracket = Text("[", font_size=48, color=WHITE)
        left_bracket.shift(RIGHT * 3.2 + UP * 0.75)
        right_bracket = Text("]", font_size=48, color=WHITE)
        right_bracket.shift(RIGHT * 4.3 + UP * 0.75)
        
        bitnet_input = MathTex(r"\begin{bmatrix} 2.1 \\ 1.4 \end{bmatrix}", color=BLUE)
        bitnet_input.shift(RIGHT * 5.5 + UP * 0.5)
        
        self.play(
            Write(bitnet_matrix), Write(left_bracket), Write(right_bracket),
            Write(bitnet_input)
        )
        
        # Show simplified operations
        operations = VGroup(
            Text("if weight = +1: output = input", font_size=16, color=GREEN),
            Text("if weight = -1: output = -input", font_size=16, color=RED),
            Text("if weight = 0: output = 0", font_size=16, color=GRAY)
        )
        operations.arrange(DOWN, buff=0.2)
        operations.shift(RIGHT * 4 + DOWN * 1.5)
        
        for op in operations:
            self.play(Write(op), run_time=0.8)
        
        # Speed comparison
        speed_comparison = Text("16x faster!", font_size=32, color=GOLD)
        speed_comparison.shift(DOWN * 3)
        self.play(Write(speed_comparison))
        self.wait(3)

    def scene5_three_line_algorithm(self):
        """Scene 5: The Three-Line Algorithm"""
        title = Text("The Three-Line Algorithm", font_size=40, color=TEAL)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Code visualization
        code_lines = [
            "if weight == 1:",
            "    output += input",
            "elif weight == -1:",
            "    output += -input", 
            "else:  # weight == 0",
            "    output = 0"
        ]
        
        code_colors = [WHITE, GREEN, WHITE, RED, WHITE, GRAY]
        
        code_group = VGroup()
        for i, (line, color) in enumerate(zip(code_lines, code_colors)):
            code_text = Text(line, font_size=24, color=color, font="monospace")
            code_text.shift(UP * (2 - i * 0.4))
            code_group.add(code_text)
        
        # Animate code appearing line by line
        for line in code_group:
            self.play(Write(line), run_time=0.8)
        
        # Decision tree visualization
        tree_root = Circle(radius=0.3, color=WHITE, fill_opacity=0.7)
        tree_root.shift(DOWN * 1.5)
        
        # Branches
        left_branch = Line(tree_root.get_center(), tree_root.get_center() + LEFT * 2 + DOWN * 1, color=GREEN)
        right_branch = Line(tree_root.get_center(), tree_root.get_center() + RIGHT * 2 + DOWN * 1, color=RED)
        center_branch = Line(tree_root.get_center(), tree_root.get_center() + DOWN * 1, color=GRAY)
        
        # Leaf nodes
        left_leaf = Circle(radius=0.25, color=GREEN, fill_opacity=0.7)
        left_leaf.move_to(tree_root.get_center() + LEFT * 2 + DOWN * 1)
        left_label = Text("+input", font_size=16, color=WHITE)
        left_label.move_to(left_leaf)
        
        right_leaf = Circle(radius=0.25, color=RED, fill_opacity=0.7)
        right_leaf.move_to(tree_root.get_center() + RIGHT * 2 + DOWN * 1)
        right_label = Text("-input", font_size=16, color=WHITE)
        right_label.move_to(right_leaf)
        
        center_leaf = Circle(radius=0.25, color=GRAY, fill_opacity=0.7)
        center_leaf.move_to(tree_root.get_center() + DOWN * 1)
        center_label = Text("0", font_size=16, color=WHITE)
        center_label.move_to(center_leaf)
        
        # Root label
        root_label = Text("weight", font_size=16, color=WHITE)
        root_label.move_to(tree_root)
        
        tree_group = VGroup(
            tree_root, left_branch, right_branch, center_branch,
            left_leaf, right_leaf, center_leaf,
            root_label, left_label, right_label, center_label
        )
        
        self.play(Create(tree_group))
        
        # Comparison with traditional multiplication
        comparison = Text("vs. Complex Floating-Point Multiplication", font_size=24, color=YELLOW)
        comparison.shift(DOWN * 3.5)
        self.play(Write(comparison))
        self.wait(3)

    def scene6_weight_quantization(self):
        """Scene 6: Weight Quantization Function Breakdown"""
        title = Text("Weight Quantization: W → {-1, 0, 1}", font_size=36, color=ORANGE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show the equation
        equation = MathTex(r"Q_w(W) = \alpha \cdot \text{RoundClip}\left(\frac{W}{\alpha + \epsilon}, -1, 1\right)")
        equation.set_color(YELLOW).scale(0.9).next_to(title, DOWN, buff=0.4)
        alpha_def = MathTex(r"\text{where } \alpha = \text{mean}(|W|)").set_color(GREEN)
        alpha_def.next_to(equation, DOWN, buff=0.4)
        self.play(Write(equation))
        self.play(Write(alpha_def))

        # Step 1: Original weights
        step1_title = Text("Step 1: Original Weights", font_size=24, color=WHITE)
        step1_title.shift(LEFT * 4 + UP * 0.5)
        self.play(Write(step1_title))
       
        original_weights = Text("[0.7, -0.3, 0.1, -0.9, 0.05]", font_size=20, color=WHITE)
        # original_weights.shift(LEFT * 4 + UP * 0.5)
        original_weights.shift(LEFT * 4 + UP * 0)
        self.play(Write(original_weights))
        
        # Step 2: Calculate alpha
        step2_title = Text("Step 2: Calculate α", font_size=24, color=GREEN)
        step2_title.shift(LEFT * 4 + DOWN*1)
        self.play(Write(step2_title))
        
        alpha_calc = Text("α = mean(|0.7|, |0.3|, |0.1|, |0.9|, |0.05|)\n= (0.7+0.3+0.1+0.9+0.05)/5 = 0.41", 
                         font_size=16, color=GREEN)
        alpha_calc.shift(LEFT * 4 + DOWN * 1.5)
        self.play(Write(alpha_calc))
        
        # Step 3: Normalize
        step3_title = Text("Step 3: Normalize W/(α+ε)", font_size=24, color=BLUE)
        step3_title.shift(RIGHT * 4 + UP * 0.5)
        self.play(Write(step3_title))
        
        normalized = Text("[1.71, -0.73, 0.24, -2.20, 0.12]", font_size=18, color=BLUE)
        normalized.shift(RIGHT * 4 + UP * 0)
        self.play(Write(normalized))
        
        # Step 4: RoundClip
        step4_title = Text("Step 4: RoundClip(-1, 1)", font_size=24, color=PURPLE)
        step4_title.shift(RIGHT * 4 + DOWN * 1)
        self.play(Write(step4_title))
        
        # Show the transformation with colors
        transformations = VGroup(
            Text("1.71 → round(1.71)=2 → clip → +1", font_size=14, color=GREEN),
            Text("-0.73 → round(-0.73)=-1 → -1", font_size=14, color=RED),
            Text("0.24 → round(0.24)=0 → 0", font_size=14, color=GRAY),
            Text("-2.20 → round(-2.20)=-2 → clip → -1", font_size=14, color=RED),
            Text("0.12 → round(0.12)=0 → 0", font_size=14, color=GRAY)
        )
        transformations.arrange(DOWN, buff=0.1)
        transformations.next_to(step4_title, DOWN, buff=0.2)
        
        for transform in transformations:
            self.play(Write(transform), run_time=0.6)
        
        # Final result
        final_result = Text("Final: [1, -1, 0, -1, 0] with α=0.41", font_size=20, color=GOLD)
        final_result.shift(DOWN * 3)
        self.play(Write(final_result))
        self.wait(3)

    def scene7_activation_quantization(self):
        """Scene 7: Activation Quantization Function"""
        title = Text("Activation Quantization: X → 4/8-bit INT", font_size=36, color=PINK)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show the equation
        equation = MathTex(r"Q_{INT8}(X) = \frac{\gamma}{127} \cdot \text{RoundClip}\left(\frac{127X}{\gamma + \epsilon}, -128, 127\right)")
        equation.set_color(YELLOW).scale(0.9).next_to(title, DOWN, buff=0.4)
        self.play(Write(equation))
        
        gamma_def = MathTex(r"\text{where } \gamma = \max(|X|)", color=GREEN)
        gamma_def.next_to(equation, DOWN, buff=0.4)
        self.play(Write(gamma_def))

        # Show the equation
        equation1 = MathTex(r"Q_{INT4}(X) = \frac{\beta}{\sqrt{7}} \cdot \text{RoundClip}\left(\frac{\sqrt{7}}{\beta + \epsilon}X, -8, 7\right)")
        equation1.set_color(YELLOW).scale(0.9).next_to(gamma_def, DOWN, buff=0.4)
        self.play(Write(equation1))
        
        gamma_def1 = MathTex(r"\text{where } \beta = mean(|X|)", color=GREEN)
        gamma_def1.next_to(equation1, DOWN, buff=0.4)
        self.play(Write(gamma_def1))
        self.wait(2)
        self.remove(equation1)
        self.remove(gamma_def1)
        
        # Step 1: Original activations
        step1_title = Text("Step 1: Original Activations", font_size=24, color=WHITE)
        step1_title.shift(LEFT * 4 + UP * 0.3)
        self.play(Write(step1_title))
        
        original_acts = Text("[2.3, -1.7, 0.8, -3.1, 1.2]", font_size=20, color=WHITE)
        original_acts.shift(LEFT * 4 + UP * 0)
        self.play(Write(original_acts))
        
        # Step 2: Find gamma (dynamic range)
        step2_title = Text("Step 2: Find γ (max range)", font_size=24, color=GREEN)
        step2_title.shift(LEFT * 4 + DOWN * 0.5)
        self.play(Write(step2_title))
        
        gamma_calc = Text("γ = max(|2.3|, |-1.7|, |0.8|, |-3.1|, |1.2|)\n= 3.1", 
                         font_size=18, color=GREEN)
        gamma_calc.shift(LEFT * 4 + DOWN * 1)
        self.play(Write(gamma_calc))
        
        # Step 3: Scale to integer range
        step3_title = Text("Step 3: Scale to [-127, 127]", font_size=24, color=BLUE)
        step3_title.shift(RIGHT * 4 + UP * 0.3)
        self.play(Write(step3_title))
        
        scaled = Text("127X/γ = 127X/3.1\n[94.4, -69.7, 32.8, -127.0, 49.2]", font_size=16, color=BLUE)
        scaled.shift(RIGHT * 4 + DOWN * 0.1)
        self.play(Write(scaled))
        
        # Step 4: RoundClip to integers
        step4_title = Text("Step 4: RoundClip to INT", font_size=24, color=PURPLE)
        step4_title.shift(RIGHT * 4 + DOWN * 1)
        self.play(Write(step4_title))
        
        final_ints = Text("[94, -70, 33, -127, 49]", font_size=18, color=PURPLE)
        final_ints.shift(RIGHT * 4 + DOWN * 1.5)
        self.play(Write(final_ints))
        
        # Dequantization formula
        dequant_title = Text("Dequantization: X_reconstructed = (γ/127) × quantized", font_size=20, color=GOLD)
        dequant_title.shift(DOWN * 2.5)
        self.play(Write(dequant_title))
        
        scale_factor = Text("Scale factor: γ/127 = 3.1/127 = 0.0244", font_size=18, color=GOLD)
        scale_factor.shift(DOWN * 3)
        self.play(Write(scale_factor))
        self.wait(3)

    def scene8_bitlinear_architecture(self):
        """Scene 8: BitLinear - The Core Building Block"""
        title = Text("BitLinear: The Core Building Block", font_size=36, color=TEAL)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Traditional vs BitLinear comparison
        trad_title = Text("Traditional Linear", font_size=24, color=RED)
        trad_title.shift(LEFT * 4 + UP * 2)
        self.play(Write(trad_title))
        
        bitlinear_title = Text("BitLinear", font_size=24, color=GREEN)
        bitlinear_title.shift(RIGHT * 4 + UP * 2)
        self.play(Write(bitlinear_title))
        
        # Traditional operation
        trad_op = Text("Output = Input × Weight + Bias\n(Full precision floating-point)", 
                      font_size=16, color=RED)
        trad_op.shift(LEFT * 4 + UP * 1)
        self.play(Write(trad_op))
        
        # BitLinear pipeline
        pipeline_steps = [
            "1. LayerNorm (full precision)",
            "2. Absmax Quantization", 
            "3. Quantized Matrix Multiply",
            "4. Dequantization (β, γ scaling)"
        ]
        
        pipeline_group = VGroup()
        for i, step in enumerate(pipeline_steps):
            step_text = Text(step, font_size=14, color=GREEN)
            step_text.shift(RIGHT * 4 + UP * (1.5 - i * 0.3))
            pipeline_group.add(step_text)
        
        for step in pipeline_group:
            self.play(Write(step), run_time=0.6)
        
        # Architecture integration
        arch_title = Text("Complete Architecture Integration", font_size=20, color=YELLOW)
        arch_title.shift(DOWN * 0.5)
        self.play(Write(arch_title))
        
        # Show replacement strategy
        replacements = VGroup(
            Text("Attention: BitLinear(Q) + BitLinear(K) + BitLinear(V)", font_size=14, color=WHITE),
            Text("FFN: BitLinear(Up) + BitLinear(Down)", font_size=14, color=WHITE),
            Text("All Linear → BitLinear (drop-in replacement)", font_size=14, color=GOLD)
        )
        replacements.arrange(DOWN, buff=0.2)
        replacements.shift(DOWN * 1.5)
        
        for replacement in replacements:
            self.play(Write(replacement), run_time=0.8)
        
        # Benefits
        benefits = Text("Benefits: 16x memory reduction, faster inference, energy efficient", 
                       font_size=18, color=GREEN)
        benefits.shift(DOWN * 3)
        self.play(Write(benefits))
        self.wait(3)

    def bitnet_architecture(self):
        def make_block(text, color=WHITE):
            return Rectangle(width=2.6, height=0.8, color=color, fill_opacity=0.15).add(Text(text, font_size=20).move_to([0, 0, 0]))

        # BitLinear block flow (left side, bottom-up)
        bitlinear_title = Text("BitLinear Flow", font_size=28, color=YELLOW).to_corner(DL)
        self.play(Write(bitlinear_title))

        bit_steps = [
            ("Input", BLUE),
            ("LayerNorm", GRAY),
            ("Absmax Quant", GREEN),
            ("1-bit Weights", YELLOW),
            ("Dequantize", ORANGE),
            ("β Scaling", PURPLE),
            ("Output", RED),
        ]
        bit_blocks = VGroup(*[make_block(label, color) for label, color in bit_steps])
        bit_blocks.arrange(UP, buff=0.4).next_to(bitlinear_title, UP, buff=0.4).shift(RIGHT * 0.5)
        self.play(Create(bit_blocks))

        # BitLinear arrows
        for i in range(len(bit_blocks) - 1):
            a = Arrow(start=bit_blocks[i].get_top(), end=bit_blocks[i+1].get_bottom(), buff=0.1, stroke_width=2)
            self.play(Create(a), run_time=0.2)

        self.wait(1)

        # BitNet architecture (right side, bottom-up)
        arch_title = Text("BitNet Architecture", font_size=28, color=ORANGE).to_corner(DR)
        self.play(Write(arch_title))

        arch_layers = [
            ("Input", BLUE),
            ("BitLinear", YELLOW),
            ("GELU", GREEN),
            ("BitLinear", YELLOW),
            ("Multi-Head Attention", PURPLE),
            ("Feed-Forward", TEAL),
            ("BitLinear", YELLOW),
            ("Output", RED)
        ]
        arch_blocks = VGroup(*[make_block(label, color) for label, color in arch_layers])
        arch_blocks.arrange(UP, buff=0.4).next_to(arch_title, UP, buff=0.4).shift(LEFT * 0.5)
        self.play(Create(arch_blocks))

        # BitNet arrows
        for i in range(len(arch_blocks) - 1):
            a = Arrow(start=arch_blocks[i].get_top(), end=arch_blocks[i+1].get_bottom(), buff=0.1, stroke_width=2)
            self.play(Create(a), run_time=0.2)

        self.wait(2)

    def transformer_architecture_with_bitlinear(self):
        # Title
        title = Text("Transformer Architecture with Bilinear Blocks", font_size=36)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)
        
        # Create the main flow
        self.create_main_architecture(title)
        self.wait(2)
        
        # Expand multihead attention
        self.expand_multihead_attention()
        self.wait(3)
        
        # Show bilinear replacement for attention
        self.show_bilinear_replacement()
        self.wait(3)
        
        # Go back to main architecture and expand FFN
        self.show_main_architecture_again()
        self.wait(2)
        
        # Expand feed forward network
        self.expand_feedforward_network()
        self.wait(3)
        
        # Show bilinear replacement for FFN
        self.show_ffn_bilinear_replacement()
        self.wait(3)

    def create_main_architecture(self, title):
        # Input
        input_box = Rectangle(width=2, height=0.8, color=BLUE, fill_opacity=0.3)
        input_text = Text("Input", font_size=24)
        input_group = VGroup(input_box, input_text)
        input_group.shift(UP * 2.5)
        
        # Multihead Attention
        mha_box = Rectangle(width=3, height=1.2, color=GREEN, fill_opacity=0.3)
        mha_text = Text("Multihead\nAttention", font_size=20)
        mha_group = VGroup(mha_box, mha_text)
        mha_group.shift(UP * 1)
        
        # Feed Forward Network
        ffn_box = Rectangle(width=3, height=1.2, color=ORANGE, fill_opacity=0.3)
        ffn_text = Text("Feed Forward\nNetwork", font_size=20)
        ffn_group = VGroup(ffn_box, ffn_text)
        ffn_group.shift(DOWN * 1)
        
        # Output
        output_box = Rectangle(width=2, height=0.8, color=RED, fill_opacity=0.3)
        output_text = Text("Output", font_size=24)
        output_group = VGroup(output_box, output_text)
        output_group.shift(DOWN * 3)
        
        # Arrows
        arrow1 = Arrow(input_group.get_bottom(), mha_group.get_top(), buff=0.1)
        arrow2 = Arrow(mha_group.get_bottom(), ffn_group.get_top(), buff=0.1)
        arrow3 = Arrow(ffn_group.get_bottom(), output_group.get_top(), buff=0.1)
        
        # Store references
        self.input_group = input_group
        self.mha_group = mha_group
        self.ffn_group = ffn_group
        self.output_group = output_group
        self.main_arrows = VGroup(arrow1, arrow2, arrow3)
        
        # Animate creation
        self.play(
            Create(input_group),
            Create(mha_group),
            Create(ffn_group),
            Create(output_group),
            Create(arrow1),
            Create(arrow2),
            Create(arrow3)
        )

    def expand_multihead_attention(self):
        # Fade out main architecture temporarily
        self.play(
            FadeOut(self.input_group),
            FadeOut(self.ffn_group),
            FadeOut(self.output_group),
            FadeOut(self.main_arrows)
        )
        
        # Move MHA to center and expand
        self.play(self.mha_group.animate.move_to(ORIGIN).scale(0.8))
        
        # Create expanded view
        # Input (top)
        exp_input = Rectangle(width=1.5, height=0.6, color=BLUE, fill_opacity=0.3)
        exp_input_text = Text("Input", font_size=18)
        exp_input_group = VGroup(exp_input, exp_input_text)
        exp_input_group.shift(UP * 2.5)
        
        # Q, K, V blocks
        q_box = Rectangle(width=1.2, height=0.8, color=YELLOW, fill_opacity=0.3)
        q_text = Text("Q", font_size=20, weight=BOLD)
        q_group = VGroup(q_box, q_text)
        q_group.shift(LEFT * 3 + UP * 1.5)
        
        k_box = Rectangle(width=1.2, height=0.8, color=YELLOW, fill_opacity=0.3)
        k_text = Text("K", font_size=20, weight=BOLD)
        k_group = VGroup(k_box, k_text)
        k_group.shift(UP * 1.5)
        
        v_box = Rectangle(width=1.2, height=0.8, color=YELLOW, fill_opacity=0.3)
        v_text = Text("V", font_size=20, weight=BOLD)
        v_group = VGroup(v_box, v_text)
        v_group.shift(RIGHT * 3 + UP * 1.5)
        
        # Attention heads (stacked vertically)
        heads = VGroup()
        
        # Main visible attention head
        main_head_box = Rectangle(width=2.5, height=0.8, color=PURPLE, fill_opacity=0.3)
        main_head_text = Text("Attention", font_size=18)
        main_head_group = VGroup(main_head_box, main_head_text)
        main_head_group.shift(UP * 0)
        heads.add(main_head_group)
        
        # Stacked heads behind (slightly offset to show stacking)
        for i in range(1, 4):  # 3 additional heads behind
            stack_head_box = Rectangle(width=2.5, height=0.8, color=PURPLE, fill_opacity=0.2)
            stack_head_group = VGroup(stack_head_box)
            stack_head_group.shift(UP * 0 + RIGHT * (i * 0.1) + DOWN * (i * 0.1))
            heads.add(stack_head_group)
        
        # Add "h Heads" label
        heads_label = Text("h Heads", font_size=14, color=PURPLE)
        heads_label.next_to(main_head_group, RIGHT, buff=0.3)
        
        # Concatenation
        concat_box = Rectangle(width=2.5, height=0.8, color=PINK, fill_opacity=0.3)
        concat_text = Text("Concatenate", font_size=18)
        concat_group = VGroup(concat_box, concat_text)
        concat_group.shift(DOWN * 1.5)
        
        # Output projection
        proj_box = Rectangle(width=2, height=0.8, color=TEAL, fill_opacity=0.3)
        proj_text = Text("Output\nProjection", font_size=16)
        proj_group = VGroup(proj_box, proj_text)
        proj_group.shift(DOWN * 2.5)
        
        # Arrows for expanded view
        # Input to Q, K, V
        arrow_q = Arrow(exp_input_group.get_bottom(), q_group.get_top(), buff=0.1)
        arrow_k = Arrow(exp_input_group.get_bottom(), k_group.get_top(), buff=0.1)
        arrow_v = Arrow(exp_input_group.get_bottom(), v_group.get_top(), buff=0.1)
        
        # Q, K, V to attention heads
        qkv_to_heads = VGroup()
        # Arrows from Q, K, V to the main attention head
        arrow_from_q = Arrow(q_group.get_bottom(), main_head_group.get_top() + LEFT * 0.6, buff=0.1, stroke_width=2)
        arrow_from_k = Arrow(k_group.get_bottom(), main_head_group.get_top(), buff=0.1, stroke_width=2)
        arrow_from_v = Arrow(v_group.get_bottom(), main_head_group.get_top() + RIGHT * 0.6, buff=0.1, stroke_width=2)
        qkv_to_heads.add(arrow_from_q, arrow_from_k, arrow_from_v)
        
        # Heads to concatenation
        heads_to_concat = Arrow(main_head_group.get_bottom(), concat_group.get_top(), buff=0.1)
        
        # Concatenation to output projection
        concat_to_proj = Arrow(concat_group.get_bottom(), proj_group.get_top(), buff=0.1)
        
        # Store references for bilinear replacement
        self.q_group = q_group
        self.k_group = k_group
        self.v_group = v_group
        self.proj_group = proj_group
        self.exp_input_group = exp_input_group
        self.heads = heads
        self.concat_group = concat_group
        
        # Animate expanded view
        self.play(FadeOut(self.mha_group))
        self.play(
            Create(exp_input_group),
            Create(q_group),
            Create(k_group),
            Create(v_group),
            Create(arrow_q),
            Create(arrow_k),
            Create(arrow_v)
        )
        self.wait(1)
        
        self.play(Create(heads), Create(heads_label))
        self.play(Create(qkv_to_heads))
        self.wait(1)
        
        self.play(Create(concat_group), Create(heads_to_concat))
        self.wait(1)
        
        self.play(Create(proj_group), Create(concat_to_proj))

    def show_bilinear_replacement(self):
        # Create bilinear blocks
        bil_q = Rectangle(width=1.2, height=0.8, color=GOLD, fill_opacity=0.5)
        bil_q_text = Text("Bilinear\nQ", font_size=14, weight=BOLD)
        bil_q_group = VGroup(bil_q, bil_q_text)
        bil_q_group.move_to(self.q_group.get_center())
        
        bil_k = Rectangle(width=1.2, height=0.8, color=GOLD, fill_opacity=0.5)
        bil_k_text = Text("Bilinear\nK", font_size=14, weight=BOLD)
        bil_k_group = VGroup(bil_k, bil_k_text)
        bil_k_group.move_to(self.k_group.get_center())
        
        bil_v = Rectangle(width=1.2, height=0.8, color=GOLD, fill_opacity=0.5)
        bil_v_text = Text("Bilinear\nV", font_size=14, weight=BOLD)
        bil_v_group = VGroup(bil_v, bil_v_text)
        bil_v_group.move_to(self.v_group.get_center())
        
        bil_proj = Rectangle(width=2, height=0.8, color=GOLD, fill_opacity=0.5)
        bil_proj_text = Text("Bilinear\nOutput", font_size=16, weight=BOLD)
        bil_proj_group = VGroup(bil_proj, bil_proj_text)
        bil_proj_group.move_to(self.proj_group.get_center())
        
        # Add replacement label
        replacement_text = Text("Bilinear Block Replacement", font_size=28, color=GOLD)
        replacement_text.to_edge(DOWN, buff=0.5)
        
        # Animate replacement
        self.play(Write(replacement_text))
        self.wait(1)
        
        # Transform blocks
        self.play(
            Transform(self.q_group, bil_q_group),
            Transform(self.k_group, bil_k_group),
            Transform(self.v_group, bil_v_group),
            Transform(self.proj_group, bil_proj_group),
        )
        
        # Add highlighting effect
        highlight_blocks = VGroup(bil_q, bil_k, bil_v, bil_proj)
        self.play(
            highlight_blocks.animate.set_stroke(GOLD, width=4),
            run_time=2
        )
        
        self.wait(2)
        
        # Final emphasis
        self.play(
            replacement_text.animate.scale(1.2).set_color(WHITE),
            run_time=1
        )

    def show_main_architecture_again(self):
        # Clear the screen
        self.clear()
        
        # Show title again
        title = Text("Transformer Architecture - Feed Forward Network", font_size=36)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Recreate main architecture with focus on FFN
        # Input
        input_box = Rectangle(width=2, height=0.8, color=BLUE, fill_opacity=0.3)
        input_text = Text("Input", font_size=24)
        input_group = VGroup(input_box, input_text)
        input_group.shift(UP * 2)
        
        # Multihead Attention (already processed)
        mha_box = Rectangle(width=3, height=1.2, color=GREEN, fill_opacity=0.3)
        mha_text = Text("Multihead\nAttention", font_size=20)
        mha_group = VGroup(mha_box, mha_text)
        mha_group.shift(UP * 0.3)
        
        # Feed Forward Network (to be expanded)
        ffn_box = Rectangle(width=3, height=1.2, color=ORANGE, fill_opacity=0.3)
        ffn_text = Text("Feed Forward\nNetwork", font_size=20)
        ffn_group = VGroup(ffn_box, ffn_text)
        ffn_group.shift(DOWN * 1.5)
        
        # Output
        output_box = Rectangle(width=2, height=0.8, color=RED, fill_opacity=0.3)
        output_text = Text("Output", font_size=24)
        output_group = VGroup(output_box, output_text)
        output_group.shift(DOWN * 3)
        
        # Arrows
        arrow1 = Arrow(input_group.get_bottom(), mha_group.get_top(), buff=0.1)
        arrow2 = Arrow(mha_group.get_bottom(), ffn_group.get_top(), buff=0.1)
        arrow3 = Arrow(ffn_group.get_bottom(), output_group.get_top(), buff=0.1)
        
        # Store references
        self.ffn_group_main = ffn_group
        self.input_group_main = input_group
        self.output_group_main = output_group
        self.mha_group_main = mha_group
        self.main_arrows_ffn = VGroup(arrow1, arrow2, arrow3)
        
        # Animate creation
        self.play(
            Create(input_group),
            Create(mha_group),
            Create(ffn_group),
            Create(output_group),
            Create(arrow1),
            Create(arrow2),
            Create(arrow3)
        )
        
        # Highlight FFN
        self.play(ffn_group.animate.set_stroke(YELLOW, width=4))

    def expand_feedforward_network(self):
        # Fade out other components
        self.play(
            FadeOut(self.input_group_main),
            FadeOut(self.mha_group_main),
            FadeOut(self.output_group_main),
            FadeOut(self.main_arrows_ffn)
        )
        
        # Move FFN to center and expand
        self.play(self.ffn_group_main.animate.move_to(ORIGIN).scale(3))
        
        # Create expanded FFN view
        # Input (top)
        ffn_input = Rectangle(width=1.5, height=0.6, color=BLUE, fill_opacity=0.3)
        ffn_input_text = Text("Input", font_size=18)
        ffn_input_group = VGroup(ffn_input, ffn_input_text)
        ffn_input_group.shift(UP * 2.5)
        
        # Up and Gate blocks
        up_box = Rectangle(width=1.5, height=0.8, color=BLUE_B, fill_opacity=0.3)
        up_text = Text("Up", font_size=20, weight=BOLD)
        up_group = VGroup(up_box, up_text)
        up_group.shift(LEFT * 2.5 + UP * 1.3)
        
        gate_box = Rectangle(width=1.5, height=0.8, color=BLUE_B, fill_opacity=0.3)
        gate_text = Text("Gate", font_size=20, weight=BOLD)
        gate_group = VGroup(gate_box, gate_text)
        gate_group.shift(RIGHT * 2.5 + UP * 1.3)
        
        # Addition operation
        add_circle = Circle(radius=0.4, color=GREEN, fill_opacity=0.3)
        add_text = Text("x", font_size=24, weight=BOLD)
        add_group = VGroup(add_circle, add_text)
        add_group.shift(UP * 0.3)
        
        # Down block
        down_box = Rectangle(width=1.5, height=0.8, color=BLUE_B, fill_opacity=0.3)
        down_text = Text("Down", font_size=20, weight=BOLD)
        down_group = VGroup(down_box, down_text)
        down_group.shift(DOWN * 1.3)
        
        # Final output
        ffn_output = Rectangle(width=1.5, height=0.6, color=RED, fill_opacity=0.3)
        ffn_output_text = Text("Output", font_size=18)
        ffn_output_group = VGroup(ffn_output, ffn_output_text)
        ffn_output_group.shift(DOWN * 2.5)
        
        # Arrows for FFN
        # Input to Up and Gate
        arrow_up = Arrow(ffn_input_group.get_bottom(), up_group.get_top(), buff=0.1)
        arrow_gate = Arrow(ffn_input_group.get_bottom(), gate_group.get_top(), buff=0.1)
        
        # Up and Gate to addition
        arrow_up_to_add = Arrow(up_group.get_bottom(), add_group.get_left(), buff=0.1)
        arrow_gate_to_add = Arrow(gate_group.get_bottom(), add_group.get_right(), buff=0.1)
        
        # Addition to Down
        arrow_add_to_down = Arrow(add_group.get_bottom(), down_group.get_top(), buff=0.1)
        
        # Down to output
        arrow_down_to_output = Arrow(down_group.get_bottom(), ffn_output_group.get_top(), buff=0.1)
        
        # Store references for bilinear replacement
        self.up_group = up_group
        self.gate_group = gate_group
        self.down_group = down_group
        self.ffn_input_group = ffn_input_group
        self.ffn_output_group = ffn_output_group
        self.add_group = add_group
        
        # Animate expanded FFN view
        self.play(FadeOut(self.ffn_group_main))
        self.play(
            Create(ffn_input_group),
            Create(up_group),
            Create(gate_group),
            Create(arrow_up),
            Create(arrow_gate)
        )
        self.wait(1)
        
        self.play(
            Create(add_group),
            Create(arrow_up_to_add),
            Create(arrow_gate_to_add)
        )
        self.wait(1)
        
        self.play(Create(down_group), Create(arrow_add_to_down))
        self.wait(1)
        
        self.play(Create(ffn_output_group), Create(arrow_down_to_output))

    def show_ffn_bilinear_replacement(self):
        # Create bilinear blocks for FFN
        bil_up = Rectangle(width=1.5, height=0.8, color=GOLD, fill_opacity=0.5)
        bil_up_text = Text("Bilinear\nUp", font_size=14, weight=BOLD)
        bil_up_group = VGroup(bil_up, bil_up_text)
        bil_up_group.move_to(self.up_group.get_center())
        
        bil_gate = Rectangle(width=1.5, height=0.8, color=GOLD, fill_opacity=0.5)
        bil_gate_text = Text("Bilinear\nGate", font_size=14, weight=BOLD)
        bil_gate_group = VGroup(bil_gate, bil_gate_text)
        bil_gate_group.move_to(self.gate_group.get_center())
        
        bil_down = Rectangle(width=1.5, height=0.8, color=GOLD, fill_opacity=0.5)
        bil_down_text = Text("Bilinear\nDown", font_size=14, weight=BOLD)
        bil_down_group = VGroup(bil_down, bil_down_text)
        bil_down_group.move_to(self.down_group.get_center())
        
        # Add replacement label
        ffn_replacement_text = Text("FFN Bilinear Block Replacement", font_size=28, color=GOLD)
        ffn_replacement_text.to_edge(DOWN, buff=0.5)
        
        # Animate replacement
        self.play(Write(ffn_replacement_text))
        self.wait(1)
        
        # Transform blocks
        self.play(
            Transform(self.up_group, bil_up_group),
            Transform(self.gate_group, bil_gate_group),
            Transform(self.down_group, bil_down_group),
        )
        
        # Add highlighting effect
        highlight_ffn_blocks = VGroup(bil_up, bil_gate, bil_down)
        self.play(
            highlight_ffn_blocks.animate.set_stroke(GOLD, width=4),
            run_time=2
        )
        
        self.wait(2)
        
        # Final emphasis
        self.play(
            ffn_replacement_text.animate.scale(1.2).set_color(WHITE),
            run_time=1
        )