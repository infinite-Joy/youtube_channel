from manim import *

class LinearQuantizationScene(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#0A0E21"
        
        # Title
        title = Text("Scale of Linear Quantization", font_size=48, color="#B8174C")
        subtitle = Text("Linear Quantization is an affine mapping of integers to real numbers", 
                        font_size=28, color=WHITE)
        subtitle.next_to(title, DOWN, buff=0.5)
        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP, buff=1)
        
        # Create the formula
        formula = MathTex(r"r = S(q - Z)", font_size=36, color=WHITE)
        formula.next_to(subtitle, DOWN, buff=0.7)
        
        # Create axes
        r_axis = NumberLine(
            x_range=[-5, 5, 1],
            length=10,
            color=WHITE,
            include_numbers=False,
            include_tip=True,
            tick_size=0.1,
        )
        r_axis.shift(UP * 1.5)
        
        q_axis = NumberLine(
            x_range=[-5, 5, 1],
            length=10,
            color=WHITE,
            include_numbers=False,
            include_tip=True,
            tick_size=0.1,
        )
        q_axis.shift(DOWN * 1.5)
        
        # Labels for axes
        r_label = Text("r", font_size=24, color=BLUE)
        r_label.next_to(r_axis, LEFT, buff=0.3)
        
        q_label = Text("q", font_size=24, color=RED)
        q_label.next_to(q_axis, LEFT, buff=0.3)
        
        # Create range indicators on r-axis
        r_min_arrow = Arrow(start=UP * 0.5, end=UP * 1.4, color=BLUE, buff=0)
        r_min_arrow.shift(LEFT * 4)
        r_min_label = Text("rₘᵢₙ", font_size=20, color=BLUE)
        r_min_label.next_to(r_min_arrow, UP, buff=0.1)
        
        r_max_arrow = Arrow(start=UP * 0.5, end=UP * 1.4, color=BLUE, buff=0)
        r_max_arrow.shift(RIGHT * 4)
        r_max_label = Text("rₘₐₓ", font_size=20, color=BLUE)
        r_max_label.next_to(r_max_arrow, UP, buff=0.1)
        
        r_zero_arrow = Arrow(start=UP * 0.5, end=UP * 1.4, color=BLUE, buff=0)
        r_zero_label = Text("0", font_size=20, color=BLUE)
        r_zero_label.next_to(r_zero_arrow, UP, buff=0.1)
        
        # Create the floating-point range rectangle
        fp_range = Rectangle(width=8, height=0.5, color=BLUE, fill_opacity=0.3)
        fp_range.move_to(r_axis.get_center())
        fp_range_label = Text("Floating-point range", font_size=20, color=BLUE)
        fp_range_label.next_to(fp_range, UP, buff=0.3)
        
        # Create range indicators on q-axis
        q_min_arrow = Arrow(start=DOWN * 0.5, end=DOWN * 1.4, color=RED, buff=0)
        q_min_arrow.shift(LEFT * 3)
        q_min_label = Text("qₘᵢₙ", font_size=20, color=RED)
        q_min_label.next_to(q_min_arrow, DOWN, buff=0.1)
        
        q_max_arrow = Arrow(start=DOWN * 0.5, end=DOWN * 1.4, color=RED, buff=0)
        q_max_arrow.shift(RIGHT * 3)
        q_max_label = Text("qₘₐₓ", font_size=20, color=RED)
        q_max_label.next_to(q_max_arrow, DOWN, buff=0.1)
        
        # Create the zero point
        z_arrow = Arrow(start=DOWN * 0.5, end=DOWN * 1.4, color=YELLOW, buff=0)
        z_arrow.shift(LEFT * 1)
        z_label = Text("Z", font_size=20, color=YELLOW)
        z_label.next_to(z_arrow, DOWN, buff=0.1)
        zero_point_label = Text("Zero point", font_size=18, color=YELLOW)
        zero_point_label.next_to(z_label, DOWN, buff=0.1)
        
        # Create quantized dots
        dots = VGroup()
        for i in range(-3, 4):
            dot = Dot(q_axis.n2p(i), color=RED)
            dots.add(dot)
        
        # Create scale factor arrow
        scale_arrow = Arrow(
            start=q_axis.n2p(0) + UP * 0.5,
            end=r_axis.n2p(0) + DOWN * 0.5,
            color=BLUE,
            buff=0.1
        )
        scale_label = MathTex(r"\times S", font_size=24, color=BLUE)
        scale_label.next_to(scale_arrow, RIGHT, buff=0.2)
        scale_title = Text("Floating-point Scale", font_size=20, color=BLUE)
        scale_title.next_to(scale_label, RIGHT, buff=0.5)
        
        # Create formulas for r_max and r_min
        formulas = VGroup(
            MathTex(r"r_{max} = S(q_{max} - Z)", font_size=28, color=WHITE),
            MathTex(r"r_{min} = S(q_{min} - Z)", font_size=28, color=WHITE),
            MathTex(r"r_{max} - r_{min} = S(q_{max} - q_{min})", font_size=28, color=WHITE),
            MathTex(r"S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}", font_size=28, color=WHITE)
        )
        formulas.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        formulas.to_edge(RIGHT, buff=1)
        formulas.shift(DOWN * 0.5)
        
        # Animation sequence
        self.play(Write(title), Write(subtitle), run_time=1.5)
        self.play(Write(formula), run_time=1)
        self.wait(0.5)
        
        # Create axes and labels
        self.play(
            Create(r_axis),
            Create(q_axis),
            Write(r_label),
            Write(q_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Create range indicators on r-axis
        self.play(
            Create(r_min_arrow),
            Write(r_min_label),
            Create(r_max_arrow),
            Write(r_max_label),
            Create(r_zero_arrow),
            Write(r_zero_label),
            run_time=1
        )
        
        # Create floating-point range
        self.play(
            Create(fp_range),
            Write(fp_range_label),
            run_time=1
        )
        
        # Create range indicators on q-axis
        self.play(
            Create(q_min_arrow),
            Write(q_min_label),
            Create(q_max_arrow),
            Write(q_max_label),
            run_time=1
        )
        
        # Create zero point
        self.play(
            Create(z_arrow),
            Write(z_label),
            Write(zero_point_label),
            run_time=1
        )
        
        # Create quantized dots with a staggered animation
        for dot in dots:
            self.play(FadeIn(dot, scale=1.2), run_time=0.2)
        
        # Create scale factor arrow
        self.play(
            Create(scale_arrow),
            Write(scale_label),
            Write(scale_title),
            run_time=1
        )
        
        # Show the formulas on the right with staggered animations
        for formula in formulas:
            self.play(Write(formula), run_time=0.8)
            self.wait(0.3)
        
        self.wait(1)
        
        # Highlight the relationship between floating point and quantized values
        highlight_dots = [dots[3], dots[4], dots[5]]  # Center dots
        highlight_lines = VGroup()
        
        for dot in highlight_dots:
            dot_x = dot.get_center()[0]
            r_value = dot_x * 1.2  # Simulating the scale factor effect
            r_point = r_axis.n2p(r_value)[0]
            
            line = DashedLine(
                start=dot.get_center(),
                end=[r_point, r_axis.get_center()[1], 0],
                color=YELLOW,
                dash_length=0.1
            )
            highlight_lines.add(line)
        
        self.play(Create(highlight_lines), run_time=1.5)
        
        # Create a final explanation box
        explanation = VGroup(
            Text("Linear Quantization maps integers (q) to", font_size=24, color=WHITE),
            Text("floating-point values (r) using scaling (S)", font_size=24, color=WHITE),
            Text("and zero-point (Z) parameters", font_size=24, color=WHITE)
        )
        explanation.arrange(DOWN, buff=0.2)
        explanation_box = SurroundingRectangle(explanation, color=YELLOW, buff=0.3)
        explanation_group = VGroup(explanation_box, explanation)
        explanation_group.to_edge(DOWN, buff=0.5)
        
        self.play(
            FadeIn(explanation_group),
            run_time=1.5
        )
        
        self.wait(2)
        
        # Final emphasis on the main formula
        final_formula = MathTex(r"r = S(q - Z)", font_size=72, color=YELLOW)
        final_formula.move_to(ORIGIN)
        
        self.play(
            FadeOut(explanation_group),
            FadeOut(highlight_lines),
            FadeOut(formulas),
            *[FadeOut(mob) for mob in self.mobjects if mob != title and mob != final_formula],
            Transform(formula, final_formula),
            run_time=2
        )
        
        self.wait(2)


class QuantizationExplanation(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#0A0E21"
        
        # Create title
        title = Text("How Linear Quantization Works", font_size=48, color="#B8174C")
        title.to_edge(UP, buff=0.5)
        
        # Create a neural network representation
        nn_layers = VGroup()
        for i in range(4):
            if i == 0 or i == 3:  # Input and output layers
                num_neurons = 3
            else:  # Hidden layers
                num_neurons = 5
                
            layer = VGroup()
            for j in range(num_neurons):
                neuron = Circle(radius=0.2, color=BLUE, fill_opacity=0.3)
                neuron.move_to([i*2, j*0.8 - (num_neurons-1)*0.4, 0])
                layer.add(neuron)
            nn_layers.add(layer)
        
        # Add connections between layers
        connections = VGroup()
        for i in range(len(nn_layers) - 1):
            layer1 = nn_layers[i]
            layer2 = nn_layers[i+1]
            for neuron1 in layer1:
                for neuron2 in layer2:
                    conn = Line(neuron1.get_center(), neuron2.get_center(), 
                               stroke_width=1, color=GREY, opacity=0.6)
                    connections.add(conn)
        
        nn = VGroup(connections, nn_layers)
        nn.scale(0.8)
        nn.to_edge(LEFT, buff=1)
        nn.shift(DOWN)
        
        # Create the weight matrix visualization
        weight_matrix = VGroup()
        matrix_rows, matrix_cols = 6, 8
        for i in range(matrix_rows):
            for j in range(matrix_cols):
                cell = Square(side_length=0.3, stroke_width=1, 
                             stroke_color=WHITE, fill_opacity=0.3)
                # Gradient fill from blue to red based on simulated weight value
                value = (i + j) / (matrix_rows + matrix_cols - 2)  # Normalized value
                color = interpolate_color(BLUE, RED, value)
                cell.set_fill(color)
                cell.move_to([j*0.35, -i*0.35, 0])
                weight_matrix.add(cell)
        
        weight_matrix.scale(1.2)
        weight_matrix.to_edge(RIGHT, buff=2)
        weight_matrix.shift(UP * 0.5)
        
        weights_label = Text("FP32 Weights", font_size=24, color=WHITE)
        weights_label.next_to(weight_matrix, UP, buff=0.3)
        
        # Create the quantized matrix
        quantized_matrix = weight_matrix.copy()
        for cell in quantized_matrix:
            # Simulate quantized colors - fewer distinct colors
            color = cell.get_fill_color()
            # Quantize the color by rounding to fewer values
            r, g, b = color[0], color[1], color[2]
            steps = 5  # Number of quantization levels
            r = round(r * steps) / steps
            g = round(g * steps) / steps
            b = round(b * steps) / steps
            cell.set_fill(rgb_to_color([r, g, b]))
        
        quantized_matrix.next_to(weight_matrix, DOWN, buff=1.5)
        quantized_label = Text("INT8 Quantized", font_size=24, color=WHITE)
        quantized_label.next_to(quantized_matrix, UP, buff=0.3)
        
        # Animation sequence
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        # Create neural network
        self.play(
            Create(nn_layers),
            run_time=1.5
        )
        self.play(
            Create(connections),
            run_time=1
        )
        
        # Show the weight matrix
        self.play(
            Write(weights_label),
            run_time=0.5
        )
        for cell in weight_matrix:
            self.play(FadeIn(cell), run_time=0.02)
        
        # Add formula in the middle
        quant_formula = MathTex(r"q = \text{round}\left(\frac{r}{S} + Z\right)", font_size=32, color=WHITE)
        quant_formula.next_to(weight_matrix, RIGHT, buff=1)
        
        dequant_formula = MathTex(r"r = S(q - Z)", font_size=32, color=WHITE)
        dequant_formula.next_to(quant_formula, DOWN, buff=0.5)
        
        arrow_down = Arrow(
            start=weight_matrix.get_bottom() + DOWN * 0.2,
            end=quantized_matrix.get_top() + UP * 0.2,
            color=YELLOW,
            buff=0.1
        )
        
        quant_process = Text("Quantization", font_size=20, color=YELLOW)
        quant_process.next_to(arrow_down, RIGHT, buff=0.3)
        
        # Show the quantization process
        self.play(
            Write(quant_formula),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Write(quant_process),
            Create(arrow_down),
            run_time=1
        )
        
        # Show the quantized matrix
        self.play(
            Write(quantized_label),
            run_time=0.5
        )
        self.play(
            FadeIn(quantized_matrix),
            run_time=1
        )
        
        # Show the dequantization formula
        self.play(
            Write(dequant_formula),
            run_time=1
        )
        
        # Create benefits box
        benefits = VGroup(
            Text("Benefits of Quantization:", font_size=28, color=YELLOW),
            Text("• 4x smaller model size", font_size=24, color=WHITE),
            Text("• Faster inference", font_size=24, color=WHITE),
            Text("• Lower memory usage", font_size=24, color=WHITE),
            Text("• Energy efficient", font_size=24, color=WHITE)
        )
        benefits.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        benefits_box = SurroundingRectangle(benefits, color=YELLOW, buff=0.3)
        benefits_group = VGroup(benefits_box, benefits)
        benefits_group.to_edge(DOWN, buff=0.5)
        
        self.play(
            FadeIn(benefits_group),
            run_time=1.5
        )
        
        self.wait(2)
        
        # Final fade to conclusion
        final_text = Text("Linear Quantization: Making LLMs Smaller & Faster", 
                         font_size=36, color="#B8174C")
        final_text.move_to(ORIGIN)
        
        self.play(
            FadeOut(benefits_group),
            FadeOut(nn),
            FadeOut(weight_matrix),
            FadeOut(quantized_matrix),
            FadeOut(weights_label),
            FadeOut(quantized_label),
            FadeOut(quant_formula),
            FadeOut(dequant_formula),
            FadeOut(arrow_down),
            FadeOut(quant_process),
            FadeOut(title),
            FadeIn(final_text),
            run_time=2
        )
        
        self.wait(2)


class QuantizationFormulasDerivation(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#0A0E21"
        
        # Create title
        title = Text("Deriving Linear Quantization Formulas", font_size=48, color="#B8174C")
        title.to_edge(UP, buff=0.5)
        
        # Starting with the basic formula
        step1 = MathTex(r"r = S(q - Z)", font_size=36, color=WHITE)
        step1.next_to(title, DOWN, buff=1)
        
        # Explain step 1
        explanation1 = Text("We start with our basic formula:", font_size=28, color=WHITE)
        explanation1.next_to(step1, UP, buff=0.3)
        
        # Step 2: Show what happens at minimum and maximum values
        step2a = MathTex(r"r_{min} = S(q_{min} - Z)", font_size=36, color=WHITE)
        step2b = MathTex(r"r_{max} = S(q_{max} - Z)", font_size=36, color=WHITE)
        
        step2_group = VGroup(step2a, step2b)
        step2_group.arrange(DOWN, buff=0.5)
        step2_group.next_to(step1, DOWN, buff=1)
        
        # Explain step 2
        explanation2 = Text("At minimum and maximum values:", font_size=28, color=WHITE)
        explanation2.next_to(step2_group, UP, buff=0.3)
        
        # Step 3: Find the range
        step3 = MathTex(r"r_{max} - r_{min} = S(q_{max} - Z) - S(q_{min} - Z)", font_size=36, color=WHITE)
        step3.next_to(step2_group, DOWN, buff=1)
        
        # Step 4: Simplify
        step4 = MathTex(r"r_{max} - r_{min} = S(q_{max} - q_{min})", font_size=36, color=WHITE)
        step4.next_to(step3, DOWN, buff=0.5)
        
        # Step 5: Solve for S
        step5 = MathTex(r"S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}", font_size=36, color=WHITE)
        step5.next_to(step4, DOWN, buff=0.5)
        
        # Highlight the final formula for S
        box = SurroundingRectangle(step5, color=YELLOW, buff=0.2)
        
        # Step 6: Now solve for Z
        step6a = MathTex(r"r_{min} = S(q_{min} - Z)", font_size=36, color=WHITE)
        step6b = MathTex(r"Z = q_{min} - \frac{r_{min}}{S}", font_size=36, color=WHITE)
        
        step6_group = VGroup(step6a, step6b)
        step6_group.arrange(DOWN, buff=0.5)
        
        # Animation sequence
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        # Step 1
        self.play(Write(explanation1), run_time=0.8)
        self.play(Write(step1), run_time=1)
        self.wait(1)
        
        # Step 2
        self.play(Write(explanation2), run_time=0.8)
        self.play(Write(step2a), Write(step2b), run_time=1.5)
        self.wait(1)
        
        # Steps 3-5
        self.play(Write(step3), run_time=1)
        self.wait(0.5)
        self.play(Write(step4), run_time=1)
        self.wait(0.5)
        self.play(Write(step5), run_time=1)
        self.wait(0.5)
        
        # Highlight final S formula
        self.play(Create(box), run_time=0.8)
        self.wait(1)
        
        # Clear and show Z derivation
        self.play(
            FadeOut(step3),
            FadeOut(step4),
            FadeOut(explanation1),
            FadeOut(explanation2),
            Transform(step1, step6a),
            FadeOut(step2_group),
            FadeOut(box),
            run_time=1.5
        )
        
        # Add Z formula
        self.play(Write(step6b), run_time=1)
        self.wait(0.5)
        
        # Highlight Z formula
        box_z = SurroundingRectangle(step6b, color=YELLOW, buff=0.2)
        self.play(Create(box_z), run_time=0.8)
        self.wait(1)
        
        # Show both final formulas
        final_formulas = VGroup(
            MathTex(r"S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}", font_size=40, color=YELLOW),
            MathTex(r"Z = q_{min} - \frac{r_{min}}{S}", font_size=40, color=YELLOW)
        )
        final_formulas.arrange(DOWN, buff=0.8)
        final_formulas.move_to(ORIGIN)
        
        self.play(
            FadeOut(step1),
            FadeOut(step6b),
            FadeOut(box_z),
            FadeOut(title),
            FadeIn(final_formulas),
            run_time=2
        )
        
        # Add box around final formulas
        final_box = SurroundingRectangle(final_formulas, color="#FF4081", buff=0.5)
        final_title = Text("Linear Quantization Parameters", font_size=36, color="#B8174C")
        final_title.next_to(final_box, UP, buff=0.3)
        
        self.play(
            Create(final_box),
            Write(final_title),
            run_time=1.5
        )
        
        self.wait(2)