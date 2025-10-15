from manim import *
import numpy as np
from scipy import stats

class DynamicQuantizationDemo(Scene):
    def construct(self):
        # Title
        title = Text("LLM Dynamic Quantization Process", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Set up layer visualization
        layer_width = 4
        layer_height = 3
        
        # Create neural network layers
        layer1 = Rectangle(width=layer_width, height=layer_height)
        layer1.set_fill(BLUE_E, opacity=0.8)
        layer1.set_stroke(BLUE, opacity=1)
        
        layer2 = Rectangle(width=layer_width, height=layer_height)
        layer2.set_fill(BLUE_D, opacity=0.8)
        layer2.set_stroke(BLUE, opacity=1)
        
        layer1.shift(LEFT * 3)
        layer2.shift(RIGHT * 3)
        
        layer1_label = Text("Hidden Layer 1", font_size=24)
        layer1_label.next_to(layer1, DOWN)
        layer2_label = Text("Hidden Layer 2", font_size=24)
        layer2_label.next_to(layer2, DOWN)
        
        # Draw arrows connecting layers
        arrow = Arrow(layer1.get_right(), layer2.get_left(), buff=0.5, color=WHITE)
        
        # Add network elements to scene
        self.play(
            Create(layer1),
            Write(layer1_label)
        )
        self.play(
            Create(arrow),
            Create(layer2),
            Write(layer2_label)
        )
        self.wait(1)

        # Step 1: Show Data Passing Through Layer 1
        step1_text = Text("Step 1: Forward Pass Through Hidden Layer", font_size=32)
        # step1_text.next_to(title, DOWN).to_edge(UP)
        step1_text.next_to(title, DOWN)
        self.play(Write(step1_text))
        
        # Animate data points passing through layer 1
        n_points = 25
        data_points = VGroup(*[
            Dot(color=YELLOW, radius=0.05) 
            for _ in range(n_points)
        ])
        
        # Position data points to left of layer 1
        for i, dot in enumerate(data_points):
            dot.move_to(layer1.get_left() + LEFT * 2 + UP * (1.5 - 3 * (i / (n_points-1))))
        
        self.play(FadeIn(data_points))
        
        # Animate data points passing through layer 1
        destinations = [
            layer1.get_center() + np.array([
                0,
                1.5 - 3 * (i / (n_points-1)),
                0
            ]) for i in range(n_points)
        ]
        
        self.play(*[
            dot.animate.move_to(dest) for dot, dest in zip(data_points, destinations)
        ])
        self.wait(0.5)
        
        # Step 2: Collect Activations
        self.play(FadeOut(step1_text))
        step2_text = Text("Step 2: Collect Layer Activations", font_size=32)
        step2_text.next_to(title, DOWN)
        self.play(Write(step2_text))
        
        # Transform dots into activations (with different values)
        # Generate a somewhat normal distribution for visualization
        activations = np.random.normal(0, 1, n_points)
        activations = sorted(activations)
        
        activation_dots = VGroup()
        for i, val in enumerate(activations):
            color = color_gradient([BLUE_E, YELLOW, RED], 101)[int((val + 3) * 16.67) % 101]
            dot = Dot(color=color, radius=0.08)
            dot.move_to(layer1.get_right() + RIGHT * 0.5 + UP * (1.5 - 3 * (i / (n_points-1))))
            activation_dots.add(dot)
        
        self.play(
            *[Transform(data_points[i], activation_dots[i]) for i in range(n_points)],
        )
        self.wait(1)
        
        # Step 3: Analyze Activation Distribution
        self.play(FadeOut(step2_text))
        step3_text = Text("Step 3: Analyze Activation Distribution", font_size=32)
        step3_text.next_to(title, DOWN)
        self.play(Write(step3_text))
        
        # Create histogram axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 7, 1],
            axis_config={"include_tip": False},
            x_length=4,
            y_length=2
        )
        axes.shift(DOWN * 1.5)
        axes_labels = axes.get_axis_labels(x_label="Activation Value", y_label="Frequency")
        
        self.play(
            Create(axes),
            Write(axes_labels)
        )
        
        # Create histogram
        hist = np.histogram(activations, bins=8, range=(-3, 3))
        histogram = VGroup()
        bin_width = 6 / 8  # Total range / number of bins
        
        for i, height in enumerate(hist[0]):
            x_pos = -3 + i * bin_width
            rect = Rectangle(
                width=bin_width,
                height=height * 0.3,
                fill_opacity=0.8,
                fill_color=BLUE,
                stroke_width=1,
                stroke_color=WHITE
            )
            rect.move_to(axes.c2p(x_pos + bin_width/2, height * 0.3 / 2))
            histogram.add(rect)
        
        self.play(FadeIn(histogram))
        self.wait(1)

        # Remove the previous layers
        self.play(FadeOut(layer2))
        self.play(FadeOut(layer2_label))
        
        # Step 4: Calculate Quantization Parameters
        self.play(FadeOut(step3_text))
        step4_text = Text("Step 4: Calculate Quantization Parameters", font_size=32)
        step4_text.next_to(title, DOWN)
        self.play(Write(step4_text))
        
        # Display min and max values
        min_val = min(activations)
        max_val = max(activations)
        
        # Calculate scale and zero point
        num_bits = 8
        q_min = 0
        q_max = 2**num_bits - 1
        
        scale = (max_val - min_val) / (q_max - q_min)
        zero_point = round(q_min - min_val / scale)
        
        # Add vertical lines for min and max
        min_line = DashedLine(
            axes.c2p(min_val, 0),
            axes.c2p(min_val, 7),
            color=RED
        )
        max_line = DashedLine(
            axes.c2p(max_val, 0),
            axes.c2p(max_val, 7),
            color=RED
        )
        
        min_label = Text(f"min = {min_val:.2f}", font_size=20, color=RED)
        min_label.next_to(min_line, DOWN)
        max_label = Text(f"max = {max_val:.2f}", font_size=20, color=RED)
        max_label.next_to(max_line, DOWN)
        
        self.play(
            Create(min_line),
            Create(max_line),
            Write(min_label),
            Write(max_label)
        )
        self.wait(1)
        
        # Display scale and zero point formulas
        scale_formula = MathTex(
            r"s = \frac{\max - \min}{q_{max} - q_{min}} = ", f"{scale:.4f}"
        )
        zero_formula = MathTex(
            r"z = round(q_{min} - \frac{\min}{s}) = ", f"{zero_point}"
        )
        
        scale_formula.to_edge(RIGHT).shift(UP * 1.5)
        zero_formula.next_to(scale_formula, DOWN, aligned_edge=LEFT)
        
        self.play(
            Write(scale_formula),
            Write(zero_formula)
        )
        self.wait(1)
        
        # Step 5: Quantize Activations
        self.play(FadeOut(step4_text))
        self.play(FadeOut(layer1))
        self.play(FadeOut(layer1_label))
        self.play(FadeOut(data_points))
        step5_text = Text("Step 5: Quantize Using Layer-Specific Parameters", font_size=32)
        step5_text.next_to(title, DOWN)
        self.play(Write(step5_text))
        
        # Formula for quantization
        quant_formula = MathTex(
            r"q = round\left(clip\left(\frac{x}{s} + z, 0, 2^8-1\right)\right)"
        )
        quant_formula.to_edge(LEFT).shift(UP * 1.5)
        
        self.play(Write(quant_formula))
        self.wait(1)
        
        # Show quantized values
        quantized_activations = [min(max(round(val/scale + zero_point), q_min), q_max) for val in activations]
        
        # Create quantized "steps"
        quant_steps = VGroup()
        unique_quant_values = sorted(list(set(quantized_activations)))
        
        for q_val in unique_quant_values:
            # Convert back to original scale for plotting
            orig_val = (q_val - zero_point) * scale
            line = DashedLine(
                axes.c2p(orig_val, 0),
                axes.c2p(orig_val, 0.5),
                color=GREEN
            )
            quant_steps.add(line)
        
        self.play(FadeIn(quant_steps))
        self.wait(1)

        # Bring back layer 2
        self.play(Unwrite(scale_formula))
        self.play(Unwrite(zero_formula))
        self.play(FadeOut(histogram))
        self.play(Uncreate(axes))
        self.play(Unwrite(axes_labels))
        self.play(Uncreate(min_line))
        self.play(Uncreate(max_line))
        self.play(Unwrite(min_label))
        self.play(Unwrite(max_label))
        self.play(FadeIn(layer2))
        self.play(FadeIn(layer2_label))
        self.wait(1)
        
        # Show data flowing to next layer
        quantized_dots = VGroup()
        for i, val in enumerate(quantized_activations):
            # This maps the quantized value back to floating point for visualization
            color = GREEN
            dot = Dot(color=color, radius=0.08)
            dot.move_to(layer2.get_left() + LEFT * 0.5 + UP * (1.5 - 3 * (i / (n_points-1))))
            quantized_dots.add(dot)
        
        self.play(*[
            data_points[i].animate.move_to(quantized_dots[i].get_center())
            for i in range(n_points)
        ])
        self.wait(0.5)
        
        self.play(*[
            data_points[i].animate.move_to(
                layer2.get_center() + np.array([0, 1.5 - 3 * (i / (n_points-1)), 0])
            ) for i in range(n_points)
        ])
        
        # Final summary
        summary_box = SurroundingRectangle(VGroup(scale_formula, zero_formula), buff=0.2)
        summary_text = Text("Each layer has its own z and s values", font_size=24)
        summary_text.next_to(summary_box, DOWN)
        
        self.play(
            Create(summary_box),
            Write(summary_text)
        )
        self.wait(1)
        
        # We will do this using the video editor
        # # Add closing message
        # closing = Text("This process repeats for each layer in the network", font_size=28)
        # closing.to_edge(DOWN)
        # self.play(Write(closing))
        # self.wait(2)