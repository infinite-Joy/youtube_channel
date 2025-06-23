"""
manim -pql qlora_quantization.py QLoRaQuantization
"""

from manim import *
import numpy as np

class QLoRaQuantization(Scene):
    def construct(self):
        self.camera.background_color = "#121212"
        
        # Title and introduction
        title = Text("QLoRA: 4-bit Quantization for LLMs", font_size=48)
        subtitle = Text("Efficient Fine-tuning with Reduced Memory", font_size=32)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Step 1: Normal Float 4-bit (NF4) Representation
        self.explain_nf4()
        
        # Step 2: Block-wise Quantization
        self.explain_block_quantization()
        
        # Step 3: Double Quantization
        self.explain_double_quantization()
        
        # Step 4: Packing 4-bit values into 8-bit storage
        self.explain_packing()
        
        # # Conclusion
        conclusion = Text("QLoRA: Memory-efficient fine-tuning for LLMs", font_size=40)
        self.play(Write(conclusion))
        self.wait(2)
        self.play(FadeOut(conclusion))

    def explain_nf4(self):
        step_title = Text("Step 1: Normal Float 4-bit (NF4)", font_size=40)
        step_title.to_edge(UP)
        self.play(Write(step_title))
        
        # Create visualization of weight distribution
        normal_dist = FunctionGraph(
            lambda x: 3 * np.exp(-(x)**2/0.5),
            x_range=[-3, 3, 0.01],
            color=BLUE
        ).scale(0.8)
        
        normal_dist_label = Text("Pretrained weights follow normal distribution", font_size=24)
        normal_dist_label.next_to(normal_dist, DOWN)
        
        self.play(Create(normal_dist), Write(normal_dist_label))
        self.wait(1)
        
        # Show NF4 quantization levels
        nf4_values = [-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0, 
                      0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
        
        # Create dots for quantization levels
        dots = VGroup()
        x_vals = np.linspace(-2.5, 2.5, 16)
        
        for i, x in enumerate(x_vals):
            dot = Dot(point=[x, 0, 0], color=RED)
            label = Text(f"{nf4_values[i]:.3f}", font_size=8).next_to(dot, DOWN, buff=0.3)
            dots.add(VGroup(dot, label))
        
        dots_title = Text("NF4 quantization levels", font_size=24)
        dots_title.next_to(dots, DOWN, buff=0.5)
        
        self.play(FadeOut(normal_dist), FadeOut(normal_dist_label))
        self.play(FadeIn(dots), Write(dots_title))
        self.wait(1)
        
        # Show the process of quantizing a weight
        example_weight = Dot(point=[1.8, 1.5, 0], color=GREEN, radius=0.15)
        example_label = Text("Original weight: 0.65", font_size=24).next_to(example_weight, UP)
        
        self.play(FadeIn(example_weight), Write(example_label))
        
        # Find closest quantization level (0.723)
        closest_dot = dots[14][0].copy().set_color(YELLOW)
        closest_label = Text("Quantized to: 0.723", font_size=24).next_to(closest_dot, UP)
        
        self.play(
            example_weight.animate.move_to(closest_dot.get_center() + UP),
            FadeOut(example_label)
        )
        self.play(
            Flash(closest_dot, color=YELLOW, line_length=0.5, flash_radius=0.5),
            Write(closest_label)
        )
        self.wait(1)
        
        bit_representation = Text("4-bit binary: 1110", font_size=24).next_to(closest_label, UP)
        self.play(Write(bit_representation))
        self.wait(1)
        
        self.play(
            FadeOut(example_weight),
            FadeOut(closest_label),
            FadeOut(bit_representation),
            FadeOut(dots),
            FadeOut(dots_title),
            FadeOut(step_title)
        )

    def explain_block_quantization(self):
        step_title = Text("Step 2: Block-wise Quantization", font_size=40)
        step_title.to_edge(UP)
        self.play(Write(step_title))
        
        # Create a weight matrix visualization
        matrix = VGroup()
        n_rows, n_cols = 6, 8
        
        # Create cells with random values
        cells = []
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                # Random value between -1 and 1
                val = np.random.uniform(-1, 1)
                color = interpolate_color(BLUE_E, RED_E, (val + 1) / 2)
                
                cell = Square(side_length=0.5)
                cell.set_fill(color, opacity=0.8)
                cell.set_stroke(WHITE, width=1)
                cell.move_to([j*0.5 - (n_cols-1)*0.25, -i*0.5 + (n_rows-1)*0.25, 0])
                row.append(cell)
            cells.append(row)
        
        flattened_cells = [cell for row in cells for cell in row]
        matrix = VGroup(*flattened_cells)
        
        matrix_label = Text("Neural network weights", font_size=24)
        matrix_label.next_to(matrix, UP)
        
        self.play(FadeIn(matrix), Write(matrix_label))
        self.wait(1)
        
        # Demonstrate block-wise quantization
        block_size = 4
        blocks = []
        
        # Group cells into blocks
        for i in range(0, n_rows, 2):
            for j in range(0, n_cols, 2):
                block_cells = []
                for bi in range(2):
                    for bj in range(2):
                        if i+bi < n_rows and j+bj < n_cols:
                            block_cells.append(cells[i+bi][j+bj])
                
                if block_cells:
                    block = VGroup(*block_cells)
                    blocks.append(block)
        
        # Add borders to highlight blocks
        block_borders = []
        for block in blocks:
            border = SurroundingRectangle(block, color=YELLOW, buff=0)
            block_borders.append(border)
        
        block_borders_group = VGroup(*block_borders)
        
        self.play(Create(block_borders_group))
        
        block_quant_label = Text("Each block quantized independently", font_size=24)
        block_quant_label.next_to(matrix, DOWN)
        
        absmax_eq = MathTex(r"scale\_factor = \max(|block|)")
        absmax_eq.next_to(block_quant_label, DOWN)
        
        self.play(Write(block_quant_label), Write(absmax_eq))
        self.wait(1)
        
        # Show quantization of one block
        focus_block = blocks[0]
        focus_border = block_borders[0]
        
        self.play(
            focus_block.animate.set_color(GREEN_A),
            focus_border.animate.set_color(GREEN).set_stroke(width=3),
            *[FadeOut(b) for b in block_borders if b != focus_border]
        )
        
        # Move the block for a closer look
        focus_block_copy = focus_block.copy()
        focus_border_copy = focus_border.copy()
        
        scale_factor = 2
        focus_group = VGroup(focus_block_copy, focus_border_copy)
        
        focus_scale_text = Text("Find absmax in block", font_size=20)
        focus_quant_text = Text("Quantize each value to nearest NF4 level", font_size=20)
        
        focus_scale_text.next_to(focus_group, DOWN)
        focus_quant_text.next_to(focus_scale_text, DOWN)
        
        self.play(
            focus_group.animate.scale(scale_factor).move_to([3, 0, 0]),
            Write(focus_scale_text)
        )
        
        # Show the quantization process
        self.play(Write(focus_quant_text))
        self.wait(1)
        
        # Return the focus back
        self.play(
            FadeOut(focus_group),
            FadeOut(focus_scale_text),
            FadeOut(focus_quant_text),
            focus_block.animate.set_color(BLUE_A),
            focus_border.animate.set_color(YELLOW).set_stroke(width=1)
        )
        
        # Clean up
        self.play(
            FadeOut(matrix),
            FadeOut(matrix_label),
            FadeOut(block_borders_group),
            FadeOut(block_quant_label),
            FadeOut(absmax_eq),
            FadeOut(step_title)
        )

    def explain_double_quantization(self):
        step_title = Text("Step 3: Double Quantization (DQ)", font_size=40)
        step_title.to_edge(UP)
        self.play(Write(step_title))
        
        # Show the scale factors from block quantization
        scales_title = Text("32-bit Scale Factors from Blocks", font_size=28)
        scales_title.next_to(step_title, DOWN, buff=0.5)
        
        # Create visualization of scale factors
        n_blocks = 5
        scale_factors = VGroup()
        
        for i in range(n_blocks):
            # Random value between 0.1 and 1.0 for scale factors
            val = np.random.uniform(0.1, 1.0)
            
            rect = Rectangle(height=0.5, width=val*3)
            rect.set_fill(BLUE, opacity=0.8)
            rect.set_stroke(WHITE, width=1)
            rect.move_to([-5, -i*0.7, 0])
            
            label = Text(f"{val:.3f}", font_size=16)
            label.next_to(rect, RIGHT)
            
            scale_block = VGroup(rect, label)
            scale_factors.add(scale_block)
        
        self.play(
            Write(scales_title),
            FadeIn(scale_factors)
        )
        self.wait(1)
        
        # # Show the double quantization process
        # dq_title = Text("Double Quantize Scale Factors (to 8-bit)", font_size=24)
        # dq_title.next_to(scale_factors, RIGHT, buff=2)
        # # dq_title.next_to(scale_factors, UP, buff=2)
        
        # arrow = Arrow(scale_factors.get_right(), dq_title.get_left(), buff=0.5)
        
        # self.play(Create(arrow), Write(dq_title))
        
        # Quantized scale factors
        quantized_scales = VGroup()
        
        for i, scale_block in enumerate(scale_factors):
            rect, label = scale_block
            
            # Simulate quantization by slightly changing the value
            quant_val = float(label.text)
            quant_val = round(quant_val * 256) / 256  # 8-bit quantization effect
            
            quant_rect = Rectangle(height=0.5, width=quant_val*3)
            quant_rect.set_fill(GREEN, opacity=0.8)
            quant_rect.set_stroke(WHITE, width=1)
            # quant_rect.move_to([dq_title.get_x(), rect.get_y(), 0])
            quant_rect.move_to([5, rect.get_y(), 0])
            
            quant_label = Text(f"{quant_val:.3f}", font_size=16)
            quant_label.next_to(quant_rect, RIGHT)
            
            quant_block = VGroup(quant_rect, quant_label)
            quantized_scales.add(quant_block)

        # Show the double quantization process
        dq_title = Text("8-bit Quantized Scale Factors", font_size=28)
        dq_title.move_to([5, scales_title.get_y(), 0])

        # Add arrow between scale factors and quantized scales
        arrow = Arrow(
            start=[-2, 0, 0],
            end=[2, 0, 0],
            buff=0.5,
            color=YELLOW
        )

        # Show memory savings
        memory_text = Text("Memory savings: 32-bit → 8-bit scale factors", font_size=24)
        memory_detail = Text("Saves ~3GB in a 65B parameter model", font_size=20)
        
        memory_text.next_to(arrow, UP, buff=0.5)
        memory_detail.next_to(memory_text, UP, buff=0.3)

        self.play(
            Write(dq_title),
            Create(arrow),
            Write(memory_text),
            Write(memory_detail)
        )

        self.wait(1)
        
        self.play(FadeIn(quantized_scales))
        self.wait(1)
        
        # # Show memory savings
        # memory_text = Text("Memory savings: 32-bit → 8-bit scale factors", font_size=24)
        # memory_text.to_edge(DOWN, buff=1)
        
        # memory_detail = Text("Saves ~3GB in a 65B parameter model", font_size=20)
        # memory_detail.next_to(memory_text, DOWN)
        
        # self.play(Write(memory_text))
        # self.play(Write(memory_detail))
        # self.wait(1)
        
        # Clean up
        self.play(
            FadeOut(scale_factors),
            FadeOut(scales_title),
            FadeOut(arrow),
            FadeOut(dq_title),
            FadeOut(quantized_scales),
            FadeOut(memory_text),
            FadeOut(memory_detail),
            FadeOut(step_title)
        )

    def explain_packing(self):
        step_title = Text("Step 4: Packing 4-bit Values into 8-bit Storage", font_size=40)
        step_title.to_edge(UP)
        self.play(Write(step_title))
        
        # Create visualization of packing 4-bit values
        bits_4_1 = Text("0101", font_size=32)
        bits_4_2 = Text("1110", font_size=32)
        
        bits_4_1.move_to([-2, 0, 0])
        bits_4_2.move_to([2, 0, 0])
        
        bits_4_1_label = Text("4-bit value 1", font_size=20)
        bits_4_2_label = Text("4-bit value 2", font_size=20)
        
        bits_4_1_label.next_to(bits_4_1, UP)
        bits_4_2_label.next_to(bits_4_2, UP)
        
        self.play(
            Write(bits_4_1), 
            Write(bits_4_2),
            Write(bits_4_1_label),
            Write(bits_4_2_label)
        )
        self.wait(1)
        
        # Show packing operation
        arrow = Arrow(ORIGIN, DOWN*2)
        
        operation_text = MathTex(r"(0101 \ll 4) | 1110 = 01011110")
        operation_text.next_to(arrow, RIGHT)
        
        self.play(Create(arrow), Write(operation_text))
        
        # Show resulting 8-bit value
        bits_8 = Text("01011110", font_size=36, color=YELLOW)
        bits_8.move_to([0, -3, 0])
        
        bits_8_label = Text("8-bit packed value", font_size=24)
        bits_8_label.next_to(bits_8, DOWN)
        
        self.play(Write(bits_8), Write(bits_8_label))
        
        # Show PyTorch implementation detail
        pytorch_note = Text("PyTorch implementation: torch.uint8 storage", font_size=20)
        pytorch_note.next_to(bits_8_label, DOWN, buff=0.5)
        
        self.play(Write(pytorch_note))
        self.wait(1)
        
        # Clean up
        self.play(
            FadeOut(bits_4_1),
            FadeOut(bits_4_2),
            FadeOut(bits_4_1_label),
            FadeOut(bits_4_2_label),
            FadeOut(arrow),
            FadeOut(operation_text),
            FadeOut(bits_8),
            FadeOut(bits_8_label),
            FadeOut(pytorch_note),
            FadeOut(step_title)
        )