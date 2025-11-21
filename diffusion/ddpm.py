from manim import *
import numpy as np


class ImageSubspaceVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Image Representation & Subspace", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Part 1: RGB Channels
        self.show_rgb_channels()
        self.wait(2)
        self.clear()
        
        # Part 2: Image Dimensions
        title.generate_target()
        title.target.scale(0.8).to_edge(UP)
        self.add(title)
        self.play(MoveToTarget(title))
        self.show_image_dimensions()
        self.wait(2)
        self.clear()
        
        # Part 3: Random vs Good Images
        self.add(title)
        self.show_image_subspace()
        self.wait(3)
    
    def show_rgb_channels(self):
        # Create three color channels
        channel_size = 1.5
        
        red_square = Square(side_length=channel_size, fill_opacity=0.7, color=RED)
        green_square = Square(side_length=channel_size, fill_opacity=0.7, color=GREEN)
        blue_square = Square(side_length=channel_size, fill_opacity=0.7, color=BLUE)
        
        channels = VGroup(red_square, green_square, blue_square)
        channels.arrange(RIGHT, buff=0.5)
        
        # Labels
        red_label = Text("Red Channel", font_size=24, color=RED).next_to(red_square, DOWN)
        green_label = Text("Green Channel", font_size=24, color=GREEN).next_to(green_square, DOWN)
        blue_label = Text("Blue Channel", font_size=24, color=BLUE).next_to(blue_square, DOWN)
        
        intensity_text = Text("Intensity: 0-255", font_size=20).next_to(channels, DOWN, buff=1)
        
        # Animate
        self.play(
            FadeIn(red_square),
            Write(red_label)
        )
        self.wait(0.5)
        self.play(
            FadeIn(green_square),
            Write(green_label)
        )
        self.wait(0.5)
        self.play(
            FadeIn(blue_square),
            Write(blue_label)
        )
        self.wait(0.5)
        self.play(Write(intensity_text))
        
        # Show combination
        combined = Square(side_length=channel_size, fill_opacity=0.8)
        combined.set_fill(WHITE)
        combined.next_to(channels, RIGHT, buff=1)
        
        plus_signs = VGroup(
            Text("+", font_size=30).move_to((red_square.get_right() + green_square.get_left()) / 2),
            Text("+", font_size=30).move_to((green_square.get_right() + blue_square.get_left()) / 2)
        )
        
        equals_sign = Text("=", font_size=30).move_to((blue_square.get_right() + combined.get_left()) / 2)
        combined_label = Text("Combined Image", font_size=24).next_to(combined, DOWN)
        
        self.play(
            Write(plus_signs),
            Write(equals_sign),
            FadeIn(combined),
            Write(combined_label)
        )
    
    def show_image_dimensions(self):
        # Show dimension breakdown
        dim_formula = MathTex(r"1 \times C \times H \times W", font_size=48)
        dim_formula.move_to(UP * 1.5)
        
        self.play(Write(dim_formula))
        self.wait()
        
        # Explanations
        explanations = VGroup(
            MathTex(r"C = \text{Channels (3 for RGB)}", font_size=32),
            MathTex(r"H = \text{Height (pixels)}", font_size=32),
            MathTex(r"W = \text{Width (pixels)}", font_size=32)
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        explanations.next_to(dim_formula, DOWN, buff=0.8)
        
        self.play(Write(explanations[0]))
        self.wait(0.5)
        self.play(Write(explanations[1]))
        self.wait(0.5)
        self.play(Write(explanations[2]))
        self.wait()
        
        # Show example dimensions
        example = Text("Example: 1×3×256×256", font_size=28, color=YELLOW)
        example.next_to(explanations, DOWN, buff=0.5)
        self.play(Write(example))
    
    def show_image_subspace(self):
        # Create a large cube representing all possible images
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6
        )
        
        # Large transparent cube for hyperspace
        hyperspace = Cube(side_length=5, fill_opacity=0.1, stroke_opacity=0.3)
        hyperspace.set_color(BLUE)
        
        hyperspace_label = Text("All Possible Images\n(Hyperspace)", font_size=24, color=BLUE)
        hyperspace_label.to_edge(LEFT).shift(UP * 2)
        
        self.play(
            Create(hyperspace),
            Write(hyperspace_label)
        )
        self.wait()
        
        # Create random noise squares
        noise_squares = VGroup()
        for _ in range(8):
            sq = Square(side_length=0.3, fill_opacity=0.8)
            # Random grayscale color
            gray_val = np.random.random()
            sq.set_fill(rgb_to_color([gray_val, gray_val, gray_val]))
            sq.move_to([
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2)
            ])
            noise_squares.add(sq)
        
        noise_label = Text("Random Noise", font_size=20, color=RED)
        noise_label.to_edge(LEFT).shift(DOWN * 0.5)
        
        self.play(
            LaggedStart(*[FadeIn(sq) for sq in noise_squares], lag_ratio=0.1),
            Write(noise_label)
        )
        self.wait()

        # Create a smaller ellipsoid for "good" images using Surface
        good_subspace = Surface(
            lambda u, v: np.array([
                1.0 * np.cos(u) * np.cos(v) + 0.5,  # x with shift
                0.75 * np.cos(u) * np.sin(v),        # y (scaled)
                0.9 * np.sin(u)                      # z (scaled)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(20, 20)
        )
        good_subspace.set_fill(GREEN, opacity=0.3)
        good_subspace.set_stroke(GREEN, width=2)
        
        good_label = Text('"Good" Images\n(Small Subspace)', font_size=24, color=GREEN)
        good_label.to_edge(RIGHT).shift(UP * 2)
        
        self.play(
            FadeIn(good_subspace),
            Write(good_label)
        )
        self.wait()
        
        # Add some "good" images inside the subspace
        good_squares = VGroup()
        for i in range(4):
            sq = Square(side_length=0.3, fill_opacity=0.9)
            # More structured colors for "good" images
            sq.set_fill([YELLOW, ORANGE, PINK, PURPLE][i])
            angle = i * PI / 2
            sq.move_to([
                0.5 + 0.6 * np.cos(angle),
                0.4 * np.sin(angle),
                0.5 * np.sin(2 * angle)
            ])
            good_squares.add(sq)
        
        self.play(
            LaggedStart(*[FadeIn(sq) for sq in good_squares], lag_ratio=0.15)
        )
        self.wait()
        
        # Final insight text
        insight = Text(
            "Finding boundaries of this subspace\nenables image generation",
            font_size=26,
            color=YELLOW
        )
        insight.to_edge(DOWN)
        
        self.play(Write(insight))
        self.wait()
        
        # Highlight the boundary
        self.play(
            good_subspace.animate.set_stroke(YELLOW, width=4),
            Flash(good_subspace, color=YELLOW, flash_radius=1.5)
        )
        self.wait(2)

# To render this animation, use:
# manim -pql script.py ImageSubspaceVisualization
# For high quality: manim -pqh script.py ImageSubspaceVisualization

from manim import *
import numpy as np

class DiffusionMCMC(Scene):
    def construct(self):
        # Title
        title = Text("Sampling from Unknown Distributions", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Part 1: The Problem
        problem_text = Text("Goal: Sample from true image distribution", font_size=28)
        problem_text.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(problem_text))
        self.wait()
        
        # Show unknown distribution
        unknown_dist = self.create_unknown_distribution()
        unknown_dist.scale(0.8).shift(LEFT * 3 + DOWN * 0.5)
        question = Text("?", font_size=60, color=RED)
        question.next_to(unknown_dist, RIGHT, buff=0.3)
        
        dist_label = Text("True Distribution\n(Unknown)", font_size=20)
        dist_label.next_to(unknown_dist, DOWN)
        
        self.play(Create(unknown_dist), Write(question), Write(dist_label))
        self.wait(2)
        
        # Part 2: The Solution - MCMC
        self.play(
            FadeOut(problem_text),
            FadeOut(question),
            unknown_dist.animate.shift(UP * 1.5).scale(0.7),
            dist_label.animate.shift(UP * 1.5).scale(0.7)
        )
        
        solution_text = Text("Solution: Markov Chain Monte Carlo (MCMC)", font_size=32, color=GREEN)
        solution_text.next_to(title, DOWN, buff=0.5)
        self.play(Write(solution_text))
        self.wait()
        
        # Key idea
        key_idea = Text("Build a Markov chain that approximates the target distribution", 
                       font_size=24, color=YELLOW)
        key_idea.next_to(solution_text, DOWN, buff=0.3)
        self.play(FadeIn(key_idea))
        self.wait(2)
        
        self.play(FadeOut(key_idea))
        
        # Part 3: Markov Chain Visualization
        self.play(
            FadeOut(unknown_dist),
            FadeOut(dist_label),
            solution_text.animate.scale(0.7).to_edge(UP, buff=0.8)
        )
        
        # Create states
        states = self.create_markov_chain()
        self.play(LaggedStart(*[Create(s) for s in states], lag_ratio=0.2))
        self.wait()
        
        # Markov property explanation
        markov_text = Text("Markovian: Current state only depends on previous state", 
                          font_size=24, color=BLUE)
        markov_text.to_edge(DOWN, buff=0.5)
        self.play(Write(markov_text))
        self.wait()
        
        # Animate transitions
        self.animate_markov_transitions(states)
        self.wait(2)
        
        self.play(FadeOut(markov_text))
        
        # Part 4: Convergence
        self.play(*[FadeOut(s) for s in states])
        
        # Show convergence concept
        convergence_group = self.show_convergence()
        self.wait(3)
        
        # Final message
        self.play(FadeOut(convergence_group))
        final_text = Text("More steps → Better approximation", 
                         font_size=36, color=GREEN)
        final_text.move_to(ORIGIN)
        self.play(Write(final_text))
        self.wait(2)
        
        self.play(FadeOut(final_text), FadeOut(solution_text))
        
    def create_unknown_distribution(self):
        # Create a mysterious distribution shape
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1, 0.2],
            x_length=4,
            y_length=2,
            axis_config={"include_tip": False, "stroke_width": 2}
        )
        
        # Complex distribution curve
        curve = axes.plot(
            lambda x: 0.3 * np.sin(2*x) + 0.5 * np.exp(-((x-5)**2)/4),
            color=BLUE,
            stroke_width=3
        )
        
        # Fill under curve
        area = axes.get_area(curve, x_range=[0, 10], color=BLUE, opacity=0.3)
        
        return VGroup(axes, curve, area)
    
    def create_markov_chain(self):
        # Create 5 states
        states = []
        positions = [LEFT * 4, LEFT * 2, ORIGIN, RIGHT * 2, RIGHT * 4]
        
        for i, pos in enumerate(positions):
            circle = Circle(radius=0.4, color=BLUE, fill_opacity=0.3)
            circle.move_to(pos)
            label = Text(f"S{i}", font_size=24)
            label.move_to(circle.get_center())
            state = VGroup(circle, label)
            states.append(state)
        
        # Add transition arrows
        for i in range(len(states) - 1):
            arrow = Arrow(
                states[i].get_right(),
                states[i+1].get_left(),
                buff=0.1,
                stroke_width=3,
                color=YELLOW
            )
            states.append(arrow)
            
            # Add probability labels
            prob = Text("p", font_size=16, color=YELLOW)
            prob.next_to(arrow, UP, buff=0.1)
            states.append(prob)
        
        # Add some backward arrows
        for i in [1, 3]:
            arrow = Arrow(
                states[i].get_left() + UP * 0.3,
                states[i-1].get_right() + UP * 0.3,
                buff=0.1,
                stroke_width=2,
                color=ORANGE
            )
            states.append(arrow)
        
        chain_group = VGroup(*states)
        chain_group.shift(UP * 0.5)
        return states
    
    def animate_markov_transitions(self, states):
        # Animate a particle moving through the chain
        particle = Dot(color=RED, radius=0.15)
        particle.move_to(states[0].get_center())
        
        self.play(FadeIn(particle))
        
        # Move through chain
        for i in range(4):
            self.play(
                particle.animate.move_to(states[i+1].get_center()),
                states[i].animate.set_fill(opacity=0.1),
                states[i+1].animate.set_fill(opacity=0.6),
                run_time=0.8
            )
            self.wait(0.3)
        
        self.play(FadeOut(particle))
    
    def show_convergence(self):
        # Show three distributions: start, middle, end
        distributions = []
        labels = ["Start\n(Far from target)", "Middle\n(Getting closer)", "End\n(Close to target)"]
        colors = [RED, YELLOW, GREEN]
        positions = [LEFT * 4, ORIGIN, RIGHT * 4]
        
        for i, (label_text, color, pos) in enumerate(zip(labels, colors, positions)):
            # Simple axes
            axes = Axes(
                x_range=[0, 5, 1],
                y_range=[0, 1, 0.5],
                x_length=2,
                y_length=1.5,
                axis_config={"include_tip": False, "stroke_width": 1}
            )
            
            # Sample distribution getting closer to target
            if i == 0:
                curve = axes.plot(lambda x: 0.5, color=color, stroke_width=2)
            elif i == 1:
                curve = axes.plot(lambda x: 0.3 + 0.2 * np.sin(x), color=color, stroke_width=2)
            else:
                curve = axes.plot(lambda x: 0.5 * np.exp(-((x-2.5)**2)/2), color=color, stroke_width=2)
            
            area = axes.get_area(curve, x_range=[0, 5], color=color, opacity=0.3)
            
            label = Text(label_text, font_size=16, color=color)
            label.next_to(axes, DOWN, buff=0.2)
            
            group = VGroup(axes, curve, area, label)
            group.move_to(pos + DOWN * 0.5)
            distributions.append(group)
        
        # Add arrows between them
        arrow1 = Arrow(distributions[0].get_right(), distributions[1].get_left(), 
                      buff=0.2, color=WHITE)
        arrow2 = Arrow(distributions[1].get_right(), distributions[2].get_left(), 
                      buff=0.2, color=WHITE)
        
        step_label = Text("More MCMC steps", font_size=20)
        step_label.next_to(arrow1, UP, buff=0.1)
        
        conv_group = VGroup(*distributions, arrow1, arrow2, step_label)
        
        self.play(LaggedStart(*[FadeIn(d) for d in distributions], lag_ratio=0.3))
        self.play(Create(arrow1), Create(arrow2), Write(step_label))
        
        return conv_group


class TransitionMatrix(Scene):
    def construct(self):
        title = Text("Transition Probability Matrix", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create transition matrix
        matrix = Matrix([
            ["p_{11}", "p_{12}", "p_{13}"],
            ["p_{21}", "p_{22}", "p_{23}"],
            ["p_{31}", "p_{32}", "p_{33}"]
        ], left_bracket="[", right_bracket="]")
        matrix.scale(0.8)
        
        # Labels
        from_label = Text("From state →", font_size=20)
        from_label.next_to(matrix, LEFT, buff=0.5)
        from_label.rotate(PI/2)
        
        to_label = Text("To state →", font_size=20)
        to_label.next_to(matrix, UP, buff=0.3)
        
        self.play(Create(matrix))
        self.play(Write(from_label), Write(to_label))
        self.wait()
        
        # Highlight a specific transition
        highlight = SurroundingRectangle(matrix.get_entries()[1], color=YELLOW)
        explanation = Text("Probability: State 1 → State 2", font_size=24, color=YELLOW)
        explanation.next_to(matrix, DOWN, buff=0.5)
        
        self.play(Create(highlight), Write(explanation))
        self.wait(2)
        
        # Show row sum = 1 property
        self.play(FadeOut(highlight), FadeOut(explanation))
        
        row_highlight = SurroundingRectangle(
            VGroup(*matrix.get_entries()[0:3]), 
            color=GREEN
        )
        row_text = Text("Each row sums to 1", font_size=24, color=GREEN)
        row_text.next_to(matrix, DOWN, buff=0.5)
        
        self.play(Create(row_highlight), Write(row_text))
        self.wait(2)

from manim import *
import numpy as np

class ForwardDiffusion(Scene):
    def construct(self):
        # Scene 1: Title
        title = Text("Forward Diffusion Process", font_size=48, gradient=(BLUE, PURPLE))
        subtitle = Text("Gradually Adding Noise to Data", font_size=32).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Scene 2: Markov Chain Concept
        title = Text("Markov Chain Concept", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        sampling_text = MathTex(r"x_0 \sim q(x)", font_size=60)
        sampling_desc = Text("Sample initial data from real distribution", 
                           font_size=24).next_to(sampling_text, DOWN)
        
        self.play(Write(sampling_text))
        self.play(FadeIn(sampling_desc))
        self.wait(2)
        self.play(FadeOut(sampling_text), FadeOut(sampling_desc), FadeOut(title))
        
        # Scene 3: Forward Diffusion Process
        title = Text("Forward Diffusion: Adding Noise Step by Step", 
                    font_size=32).to_edge(UP)
        self.play(Write(title))
        
        # Create circles representing states
        circles = VGroup()
        labels = VGroup()
        
        positions = [LEFT * 5, LEFT * 2, RIGHT * 1, RIGHT * 4]
        names = [r"x_T", r"x_t", r"x_{t-1}", r"x_0"]
        
        for pos, name in zip(positions, names):
            circle = Circle(radius=0.6, color=PINK, fill_opacity=0.3).move_to(pos)
            label = MathTex(name).move_to(circle)
            circles.add(circle)
            labels.add(label)
        
        # Create arrows with probability labels
        arrows = VGroup()
        prob_labels = VGroup()
        
        arrow_labels_text = [r"...", r"p_{\theta}(x_{t-1} | x_t)", r"..."]
        
        for i in range(len(circles) - 1):
            if i == 1:
                arrow = Arrow(circles[i].get_right(), circles[i+1].get_left(), 
                            buff=0.15, color=YELLOW)
            else:
                arrow = DashedLine(circles[i].get_right(), circles[i+1].get_left(), 
                                 buff=0.15).add_tip()
            arrows.add(arrow)
            
            if i == 1:
                prob_label = MathTex(arrow_labels_text[i], font_size=28)
                prob_label.next_to(arrow, UP, buff=0.1)
            elif i == 2:
                prob_label = MathTex(arrow_labels_text[i], font_size=28)
                prob_label.next_to(arrow, UP, buff=0.1)
            else:
                prob_label = MathTex(arrow_labels_text[i], font_size=28)
                prob_label.next_to(arrow, UP, buff=0.1)
            
            prob_labels.add(prob_label)
        
        # Add backward arrow from x_{t-1} to x_t
        #backward_arrow = Arrow(circles[2].get_left() + UP * 0.3, 
        #                      circles[1].get_left() + UP * 0.3, 
        #                      buff=0.15, color=GREEN)
        backward_arrow = Arrow(circles[2].get_left() + DOWN, 
                              circles[1].get_left() + DOWN, 
                              color=GREEN)
        backward_label = MathTex(r"q(x_t | x_{t-1})", font_size=28, color=GREEN)
        backward_label.next_to(backward_arrow, DOWN, buff=0.1)
        
        # Animate creation
        self.play(*[Create(c) for c in circles], 
                 *[Write(l) for l in labels])
        self.wait(1)
        
        self.play(*[Create(a) for a in arrows],
                 *[Write(p) for p in prob_labels])
        self.wait(1)
        
        self.play(Create(backward_arrow), Write(backward_label))

        # Create visual representation of data becoming noisy
        # Load an actual image from disk
        # Replace 'path/to/your/image.png' with your actual image path
        start_x = 4
        timesteps = len(circles)
        spacing = 2.0

        clean_image = ImageMobject(r"D:\youtube\manimations\diffusion\240_F_1434102712_XDZ4XBljlu4Ico4AE9HOXyGw3hmPsovt.jpg")  # Change this to your image path
        clean_image.scale(0.8)
        clean_image.move_to([start_x, -2.0, 0])

        # Create noisy version by overlaying random pixels
        noisy_dots = VGroup()
        np.random.seed(42)
        for _ in range(200):
            x = start_x - timesteps * spacing - 1 + np.random.uniform(-0.5, 0.5)
            y = -2.0 + np.random.uniform(-0.5, 0.5)
            dot = Dot(
                point=[x, y, 0],
                radius=0.015,
                color=random_bright_color()
            )
            noisy_dots.add(dot)
        
        # Show data transformation
        self.play(
            FadeIn(clean_image, scale=0.8),
            run_time=1.5
        )
        
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(dot, scale=0.5) for dot in noisy_dots],
                lag_ratio=0.005
            ),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Add noise visualization
        noise_text = Text("Pure Noise", font_size=20).next_to(circles[0], DOWN)
        clean_text = Text("Clean Image", font_size=20).next_to(circles[3], DOWN)
        
        self.play(FadeIn(noise_text), FadeIn(clean_text))
        self.wait(100)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # Scene 4: Joint Probability Formula
        title = Text("Joint Probability Distribution", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        markov_text = Text("Markov Process: Memoryless Property", 
                          font_size=28).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(markov_text))
        self.wait(2)
        
        # Joint probability formula
        joint_prob = MathTex(
            r"q(x_1, ..., x_T) = \prod_{t=1}^{T} q(x_t | x_{t-1})",
            font_size=80
        )
        
        self.play(Write(joint_prob))
        self.wait(2)
        
        # Box around formula
        box = SurroundingRectangle(joint_prob, color=BLUE, buff=0.3)
        self.play(Create(box))
        
        # Add explanation
        explanation = Text("Forward Diffusion Process", 
                          font_size=28, color=YELLOW).next_to(box, DOWN, buff=0.5)
        self.play(FadeIn(explanation))
        self.wait(3)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # Scene 5: Transition Kernel
        title = Text("Transition Kernel: Gaussian Distribution", 
                    font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Normal distribution reminder
        normal_def = MathTex(
            r"\mathcal{N}(\mu, \sigma^2)",
            font_size=60
        ).shift(UP * 2)
        
        normal_text = Text("Mean μ, Variance σ²", font_size=24).next_to(normal_def, DOWN)
        
        self.play(Write(normal_def))
        self.play(FadeIn(normal_text))
        self.wait(2)

        # Create Gaussian distribution graph
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": False, "font_size": 20},
            tips=False
        ).shift(DOWN * 1.2)
        
        # Gaussian function
        def gaussian(x, mu=0, sigma=1):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        # Plot the Gaussian curve
        gaussian_curve = axes.plot(
            lambda x: gaussian(x, mu=0, sigma=1),
            x_range=[-4, 4],
            color=BLUE,
            stroke_width=3
        )
        
        # Fill under curve
        area = axes.get_area(
            gaussian_curve,
            x_range=[-4, 4],
            color=BLUE,
            opacity=0.3
        )
        
        # Add mean line
        mean_line = DashedLine(
            axes.c2p(0, 0),
            axes.c2p(0, gaussian(0)),
            color=GREEN,
            stroke_width=3
        )
        mean_label = MathTex(r"\mu", color=GREEN, font_size=30).next_to(mean_line, DOWN, buff=0.1)
        
        # Add sigma markers
        sigma_left = axes.c2p(-1, 0)
        sigma_right = axes.c2p(1, 0)
        
        sigma_line_left = DashedLine(
            sigma_left,
            axes.c2p(-1, gaussian(-1)),
            color=RED,
            stroke_width=2
        )
        sigma_line_right = DashedLine(
            sigma_right,
            axes.c2p(1, gaussian(1)),
            color=RED,
            stroke_width=2
        )
        
        sigma_brace = BraceBetweenPoints(sigma_left, sigma_right, direction=DOWN, color=RED)
        sigma_label = MathTex(r"\sigma", color=RED, font_size=28).next_to(sigma_brace, DOWN, buff=0.1)
        
        # Animate the graph
        self.play(Create(axes), run_time=1)
        self.play(Create(gaussian_curve), Create(area), run_time=1.5)
        self.play(Create(mean_line), Write(mean_label))
        self.wait(0.5)
        self.play(
            Create(sigma_line_left), 
            Create(sigma_line_right),
            Create(sigma_brace),
            Write(sigma_label)
        )
        graph_objs = [axes, area, gaussian_curve, mean_line, mean_label, sigma_line_left, sigma_line_right, sigma_brace, sigma_label]
        self.wait(2)

        self.play(FadeOut(normal_def), FadeOut(normal_text))
        self.play(*[FadeOut(mob) for mob in graph_objs])
        self.wait(1)

        # Transition kernel formula
        transition = MathTex(
            r"q(x_t | x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)",
            font_size=60
        )
        
        self.play(Write(transition))
        self.wait(2)
        
        # Highlight components
        mean_box = SurroundingRectangle(transition[0][16:27], color=GREEN)
        var_box = SurroundingRectangle(transition[0][28:], color=RED)
        
        mean_label = Text("Mean", font_size=24, color=GREEN).next_to(mean_box, DOWN, buff=0.5)
        var_label = Text("Variance", font_size=24, color=RED).next_to(var_box, DOWN, buff=0.5)
        
        self.play(Create(mean_box), FadeIn(mean_label))
        self.wait(1)
        self.play(Create(var_box), FadeIn(var_label))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in [mean_box, var_box, mean_label, var_label]])
        
        # Beta schedule explanation
        beta_text = Text("β: Noise Variance Schedule", font_size=28, color=YELLOW)
        beta_range = MathTex(r"0 < \beta_1 < \beta_2 < ... < \beta_T < 1", 
                           font_size=32).next_to(beta_text, DOWN)
        
        group = VGroup(beta_text, beta_range).next_to(transition, DOWN, buff=0.8)
        
        self.play(FadeIn(beta_text))
        self.play(Write(beta_range))
        self.wait(3)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # Additional transition kernel details
        title = Text("Understanding the Transition Kernel", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        formula = MathTex(
            r"q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)",
            font_size=60
        ).shift(UP * 1.5)
        
        self.play(Write(formula))
        self.wait(1)
        
        # Explain each component
        explanation1 = Text("Mean shows relation with previous state", font_size=24).shift(UP * 0.5)
        self.play(FadeIn(explanation1))
        self.wait(2)
        self.play(FadeOut(explanation1))
        
        explanation2 = Text("Variance adds noise", font_size=24).shift(DOWN * 0.5)
        self.play(FadeIn(explanation2))
        self.wait(2)
        self.play(FadeOut(explanation2))
        
        explanation3 = Text("Identity matrix: same variance in all dimensions", 
                          font_size=24).shift(DOWN * 1.5)
        self.play(FadeIn(explanation3))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # Scene 6: Variance Schedules
        title = Text("Variance Schedule Strategies", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[0, 1000, 200],
            y_range=[-8, 4, 2],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": True, "font_size": 24},
            x_axis_config={"numbers_to_include": [0, 200, 400, 600, 800, 1000]},
            y_axis_config={"numbers_to_include": [-8, -6, -4, -2, 0, 2, 4]},
        ).scale(0.7).shift(DOWN * 0.5)
        
        x_label = axes.get_x_axis_label(MathTex(r"Timestep \ t", font_size=28))
        y_label = axes.get_y_axis_label(MathTex(r"\log(SNR)", font_size=28), edge=LEFT)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Define different schedules
        T = 1000
        
        schedules = [
            ("Linear", RED, lambda t: 3 - 6 * t / T),
            ("Fibonacci", GOLD, lambda t: 3.5 - 7 * (t / T) ** 0.7),
            ("Cosine", GREEN, lambda t: 4 - 12 * (t / T) ** 1.5),
            ("Sigmoid", TEAL, lambda t: 3 - 6 / (1 + np.exp(-10 * (t / T - 0.5)))),
            ("Exponential", BLUE, lambda t: 4 * np.exp(-3 * t / T) - 1),
        ]
        
        curves = VGroup()
        legend_items = VGroup()
        
        for i, (name, color, func) in enumerate(schedules):
            # Create curve
            curve = axes.plot(
                lambda t: func(t),
                x_range=[1, T],
                color=color,
                stroke_width=3
            )
            curves.add(curve)
            
            # Create legend item
            legend_x = 5.5
            legend_y = 2.5 - i * 0.5
            line = Line(LEFT * 0.3, RIGHT * 0.3, color=color, stroke_width=4)
            line.move_to([legend_x, legend_y, 0])
            text = Text(name, font_size=20, color=color).next_to(line, RIGHT, buff=0.2)
            legend_items.add(VGroup(line, text))
        
        # Animate curves sequentially
        for i, (curve, legend_item) in enumerate(zip(curves, legend_items)):
            self.play(Create(curve), FadeIn(legend_item), run_time=0.8)
            self.wait(0.3)
        
        # Highlight cosine schedule
        cosine_highlight = Text("Cosine Schedule: Better Performance", 
                               font_size=24, color=GREEN).to_edge(DOWN)
        self.play(FadeIn(cosine_highlight))
        self.wait(2)
        
        # Original DDPM parameters
        ddpm_text = MathTex(
            r"\text{DDPM: } \beta_1 = 10^{-4} \text{ to } \beta_T = 0.02 \text{ (Linear)}",
            font_size=24
        ).next_to(cosine_highlight, UP)
        
        self.play(FadeIn(ddpm_text))
        self.wait(3)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # Final summary
        summary = Text("Forward Diffusion Complete!", font_size=48, 
                      gradient=(BLUE, PURPLE))
        self.play(Write(summary))
        self.wait(2)
        self.play(FadeOut(summary))


from manim import *

class ForwardDiffusionReparameterization(Scene):
    def construct(self):
        # Title
        title = Text("Tractable Closed Form Sampling", font_size=40, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(2)
        
        # Section 1: The Challenge
        self.challenge_section()
        self.play(FadeOut(title))
        
        # Section 2: Reparameterization Trick Introduction
        self.reparam_intro()
        
        # Section 3: Applying to our case
        eq8, section_title = self.apply_reparam()
        
        # Section 4: Recursive expansion (continues from apply_reparam)
        eq_final, section_title = self.recursive_expansion(eq8, section_title)
        
        # Section 5: Combining normal distributions
        combined = self.combine_normals(eq_final, section_title)
        
        # Section 6: Final closed form
        self.final_form(combined)

    def challenge_section(self):
        challenge = VGroup(
            Text("Challenge: Sample at t=500?", font_size=32),
            Text("Need to apply q 500 times?", font_size=28),
            Text("Dynamic programming = sequential training", font_size=28)
        ).arrange(DOWN, buff=0.4)
        challenge.move_to(ORIGIN)
        
        self.play(FadeIn(challenge[0]))
        self.wait(1.5)
        self.play(FadeIn(challenge[1]))
        self.wait(1.5)
        self.play(FadeIn(challenge[2]))
        self.wait(2)
        
        solution = Text("Better method: Reparameterization Trick!", 
                       font_size=32, color=GREEN).next_to(challenge, DOWN, buff=0.6)
        self.play(Write(solution))
        self.wait(2)
        self.play(FadeOut(challenge), FadeOut(solution))

    def reparam_intro(self):
        intro_text = Text("Reparameterization Trick for Normal Distribution", 
                         font_size=32, color=YELLOW)
        intro_text.to_edge(UP)
        self.play(Write(intro_text))
        self.wait(1)
        
        # Original distribution
        original = MathTex(r"z \sim \mathcal{N}(\mu, \sigma^2)").shift(UP*2.5)
        self.play(Write(original))
        self.wait(1)
        
        # Create axes for visualization
        axes = Axes(
            x_range=[-4, 8, 1],
            y_range=[0, 0.5, 0.1],
            x_length=8,
            y_length=3,
            axis_config={"color": BLUE, "include_tip": False},
        ).shift(DOWN*0.5)
        
        # Parameters for original distribution
        mu_val = 2
        sigma_val = 1.5
        
        # Original Gaussian curve
        original_gauss = axes.plot(
            lambda x: (1/(sigma_val * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu_val)/sigma_val)**2),
            color=RED,
            x_range=[-2, 6]
        )
        
        # Labels for original distribution
        mu_label = MathTex(r"\mu", color=RED).next_to(axes.c2p(mu_val, 0), DOWN)
        sigma_label = MathTex(r"\sigma", color=RED).next_to(axes.c2p(mu_val + sigma_val, 0.15), UP)
        
        # Vertical line at mu
        mu_line = DashedLine(
            axes.c2p(mu_val, 0),
            axes.c2p(mu_val, 0.27),
            color=RED
        )
        
        self.play(Create(axes))
        self.play(Create(original_gauss), Create(mu_line))
        self.play(Write(mu_label), Write(sigma_label))
        self.wait(2)
        
        # Show the transformation
        transform_text = Text("Transform to standard normal", font_size=24, color=ORANGE)
        transform_text.next_to(axes, DOWN, buff=0.5)
        self.play(Write(transform_text))
        self.wait(1)
        
        # Standard normal curve (mu=0, sigma=1)
        standard_gauss = axes.plot(
            lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2),
            color=GREEN,
            x_range=[-4, 4]
        )
        
        # Labels for standard distribution
        zero_label = MathTex(r"0", color=GREEN).next_to(axes.c2p(0, 0), DOWN)
        one_label = MathTex(r"1", color=GREEN).next_to(axes.c2p(1, 0.24), UP)
        
        # Vertical line at 0
        zero_line = DashedLine(
            axes.c2p(0, 0),
            axes.c2p(0, 0.4),
            color=GREEN
        )
        
        # Animate the transformation
        self.play(
            Transform(original_gauss, standard_gauss),
            Transform(mu_line, zero_line),
            Transform(mu_label, zero_label),
            Transform(sigma_label, one_label),
        )
        self.wait(2)
        
        # Show epsilon notation
        epsilon_notation = MathTex(r"\epsilon \sim \mathcal{N}(0,1)", color=GREEN)
        epsilon_notation.next_to(transform_text, DOWN, buff=0.3)
        self.play(Write(epsilon_notation))
        self.wait(2)
        
        # Clear visualization and show the formula
        self.play(
            FadeOut(axes), FadeOut(original_gauss), FadeOut(mu_line),
            FadeOut(mu_label), FadeOut(sigma_label), FadeOut(transform_text),
            FadeOut(epsilon_notation)
        )
        
        arrow = Arrow(LEFT, RIGHT, color=BLUE).next_to(original, DOWN, buff=0.5)
        self.play(Create(arrow))
        
        # Reparameterized form
        reparam = MathTex(r"z = \mu + \sigma\epsilon \text{ where } \epsilon \sim \mathcal{N}(0,1)")
        reparam.next_to(arrow, DOWN, buff=0.5)
        self.play(Write(reparam))
        self.wait(1)
        
        # Show the equivalence step by step
        step1 = MathTex(r"z - \mu = \sigma\epsilon").next_to(reparam, DOWN, buff=0.6)
        step2 = MathTex(r"\frac{z - \mu}{\sigma} = \epsilon").next_to(step1, DOWN, buff=0.3)
        step3 = MathTex(r"\epsilon = \frac{z - \mu}{\sigma} \sim \mathcal{N}(0,1)", color=GREEN).next_to(step2, DOWN, buff=0.3)
        
        self.play(Write(step1))
        self.wait(1)
        self.play(Write(step2))
        self.wait(1)
        self.play(Write(step3))
        self.wait(2)
        
        self.play(
            FadeOut(original), FadeOut(arrow), FadeOut(reparam), 
            FadeOut(intro_text), FadeOut(step1), FadeOut(step2), FadeOut(step3)
        )

    def apply_reparam(self):
        section_title = Text("Applying to Diffusion Process", font_size=32, color=YELLOW)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        
        # Equation 5 reference
        eq5_ref = MathTex(
            r"q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)"
        ).scale(0.8).shift(UP*2)
        self.play(Write(eq5_ref))
        self.wait(2)
        
        # Equation 6
        eq6 = MathTex(
            r"x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}"
        ).next_to(eq5_ref, DOWN, buff=0.5)
        self.play(TransformFromCopy(eq5_ref, eq6))
        self.wait(2)
        
        # Fade out eq5
        self.play(FadeOut(eq5_ref))
        self.play(eq6.animate.shift(UP*1))
        
        # Introduce alpha definition
        alpha_def = MathTex(r"\alpha_t = 1 - \beta_t").next_to(eq6, DOWN, buff=0.8)
        alpha_box = SurroundingRectangle(alpha_def, color=BLUE, buff=0.15)
        self.play(Write(alpha_def))
        self.play(Create(alpha_box))
        self.wait(2)
        
        # Highlight the beta terms in eq6 that will be replaced
        eq6_highlight = MathTex(
            r"x_t = \sqrt{"
            r"1-\beta_t"
            r"} x_{t-1} + \sqrt{"
            r"\beta_t"
            r"} \epsilon_{t-1}"
        ).move_to(eq6)
        #eq6_highlight[1].set_color(YELLOW)
        #eq6_highlight[3].set_color(YELLOW)
        
        self.play(Transform(eq6, eq6_highlight))
        self.wait(1.5)
        
        # Show the substitution step by step
        # First: 1 - beta_t -> alpha_t
        eq6_step1 = MathTex(
            r"x_t = \sqrt{"
            r"\alpha_t"
            r"} x_{t-1} + \sqrt{"
            r"\beta_t"
            r"} \epsilon_{t-1}"
        ).move_to(eq6)
        # eq6_step1[1].set_color(GREEN)
        # eq6_step1[3].set_color(YELLOW)
        
        self.play(Transform(eq6, eq6_step1))
        self.wait(1.5)
        
        # Second: beta_t -> 1 - alpha_t
        eq8 = MathTex(
            r"x_t = \sqrt{"
            r"\alpha_t"
            r"} x_{t-1} + \sqrt{"
            r"1-\alpha_t"
            r"} \epsilon_{t-1}"
        ).move_to(eq6)
        # eq8[1].set_color(GREEN)
        # eq8[3].set_color(GREEN)
        
        self.play(Transform(eq6, eq8))
        self.wait(2)
        
        # Remove color highlighting
        eq8_final = MathTex(
            r"x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}"
        ).move_to(eq6)
        
        self.play(Transform(eq6, eq8_final))
        self.wait(1)

        # Fade out alpha definition and its box, but keep section title
        self.play(FadeOut(alpha_def), FadeOut(alpha_box))
        
        # Return the equation object for recursive_expansion to use
        return eq6, section_title
        
    def recursive_expansion(self, eq8, section_title):
        # Update section title
        new_title = Text("Recursive Expansion", font_size=32, color=YELLOW)
        new_title.to_edge(UP)
        self.play(Transform(section_title, new_title))
        self.wait(1)
        
        # Move eq8 to top for more space
        self.play(eq8.animate.next_to(new_title, DOWN, buff=1.0))
        self.wait(1)
        
        # Highlight x_{t-1} that we'll substitute
        eq8_highlight = MathTex(
            r"x_t = \sqrt{\alpha_t} "
            r"x_{t-1}"
            r" + \sqrt{1-\alpha_t} \epsilon_{t-1}"
        ).move_to(eq8)
        # eq8_highlight[1].set_color(YELLOW)
        self.play(Transform(eq8, eq8_highlight))
        self.wait(1.5)
        
        # Show substitution formula below
        subst_formula = MathTex(
            r"x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}} \epsilon_{t-2}"
        ).scale(0.75).next_to(eq8, DOWN, buff=0.6)
        subst_box = SurroundingRectangle(subst_formula, color=ORANGE, buff=0.1)
        self.play(Write(subst_formula))
        self.play(Create(subst_box))
        self.wait(2)
        
        # Morph to show substitution (equation 9)
        eq9 = MathTex(
            r"x_t = \sqrt{\alpha_t} ("
            r"\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}} \epsilon_{t-2}"
            r") + \sqrt{1-\alpha_t} \epsilon_{t-1}"
        ).move_to(eq8)
        # eq9[1].set_color(ORANGE)
        
        self.play(
            FadeOut(subst_formula), 
            FadeOut(subst_box),
            Transform(eq8, eq9)
        )
        self.wait(2)
        
        # Expand/distribute (equation 10) - morph to show multiplication
        eq10 = MathTex(
            r"x_t = "
            r"\sqrt{\alpha_t\alpha_{t-1}} x_{t-2}"
            r" + "
            r"\sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-2}"
            r" + \sqrt{1-\alpha_t} \epsilon_{t-1}"
        ).move_to(eq8)
        # eq10[1].set_color(BLUE)
        # eq10[3].set_color(PURPLE)
        # eq10[4].set_color(PURPLE)
        
        self.play(Transform(eq8, eq10))
        self.wait(2)
        
        # Highlight the two epsilon terms
        # epsilon_box1 = SurroundingRectangle(eq10[3], color=PURPLE, buff=0.05)
        # epsilon_box2 = SurroundingRectangle(eq10[4], color=PURPLE, buff=0.05)
        # self.play(Create(epsilon_box1), Create(epsilon_box2))
        # self.wait(1)
        
        question = Text("Can we combine these ε terms?", font_size=28, color=ORANGE)
        question.next_to(eq8, DOWN, buff=0.8)
        self.play(Write(question))
        self.wait(2)
        
        # self.play(FadeOut(epsilon_box1), FadeOut(epsilon_box2), FadeOut(question))
        self.play(FadeOut(question))
        
        # Return the equation for next section
        return eq8, section_title

    def combine_normals(self, eq_from_prev, section_title):
        # Update section title
        new_title = Text("Combining Normal Distributions", font_size=32, color=YELLOW)
        new_title.to_edge(UP)
        self.play(Transform(section_title, new_title))
        self.wait(1)
        
        # Clear the previous equation
        self.play(FadeOut(eq_from_prev))
        
        # Two independent normals
        x_dist = MathTex(r"X \sim \mathcal{N}(0, \alpha_t(1-\alpha_{t-1})I)").scale(0.8)
        y_dist = MathTex(r"Y \sim \mathcal{N}(0, (1-\alpha_t)I)").scale(0.8)
        dists = VGroup(x_dist, y_dist).arrange(DOWN, buff=0.4).shift(UP)
        self.play(Write(x_dist))
        self.wait(1)
        self.play(Write(y_dist))
        self.wait(1)
        
        # Sum rule
        sum_rule = MathTex(r"Z = X + Y \sim \mathcal{N}(\mu_x + \mu_y, \sigma_x^2 + \sigma_y^2)")
        sum_rule.scale(0.75).next_to(dists, DOWN, buff=0.6)
        self.play(Write(sum_rule))
        self.wait(2)
        
        # Calculate variance sum
        var_calc1 = MathTex(
            r"\sigma_x^2 + \sigma_y^2 &= \alpha_t(1-\alpha_{t-1}) + 1 - \alpha_t\\"
        ).next_to(sum_rule, DOWN, buff=0.5)

        # move the var_calc1 to the top
        self.play(Write(var_calc1))
        self.play(FadeOut(dists), FadeOut(sum_rule))
        self.play(var_calc1.animate.next_to(new_title, DOWN, buff=1.0))
        self.wait(1)

        var_calc2 = MathTex(
            r"\sigma_x^2 + \sigma_y^2 &= \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t\\"
        )
        var_calc3 = MathTex(
            r"\sigma_x^2 + \sigma_y^2 &= 1 - \alpha_t\alpha_{t-1}"
        )
        self.wait(1)

        # self.play(Write(var_calc2))
        self.play(ReplacementTransform(var_calc1, var_calc2))
        self.wait(1)
        # self.play(Write(var_calc3))
        self.play(ReplacementTransform(var_calc2, var_calc3))
        self.wait(2)
        
        # Combined result
        combined = MathTex(
            r"\therefore \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-2} + \sqrt{1-\alpha_t} \epsilon_{t-1} = \sqrt{1-\alpha_t\alpha_{t-1}} \epsilon_{t-2}"
        ).next_to(var_calc3, DOWN, buff=0.5)
        combined.set_color(GREEN)
        self.play(Write(combined))
        self.wait(3)
        
        ## self.play(FadeOut(var_calc3), FadeOut(combined), FadeOut(section_title))
        self.play(FadeOut(var_calc3))
        self.wait(1)

        combined.set_color(WHITE)
        self.play(combined.animate.next_to(section_title, DOWN, buff=1.0))
        self.wait(1)
        self.play(FadeOut(section_title))
        self.wait(1)

        return combined

    def final_form(self, combined):
        section_title = Text("Final Closed Form", font_size=32, color=YELLOW)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        
        # Recursive pattern
        pattern = MathTex(
            r"x_t = \sqrt{\alpha_t\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}} \epsilon_{t-2}"
        ).scale(0.75).shift(UP*2)
        # self.play(Write(pattern))
        self.play(ReplacementTransform(combined, pattern))
        self.wait(2)
        
        # Continue recursion
        dots = MathTex(r"\vdots").next_to(pattern, DOWN)
        self.play(Write(dots))
        self.wait(1)
        
        # To x_0
        to_x0 = MathTex(
            r"x_t = \sqrt{\alpha_t\alpha_{t-1}...\alpha_1} x_0 + \sqrt{1-\alpha_t\alpha_{t-1}...\alpha_1} \epsilon_0"
        ).scale(0.7).next_to(dots, DOWN, buff=0.4)
        self.play(Write(to_x0))
        self.wait(2)
        
        # Define bar alpha
        bar_alpha = MathTex(r"\overline{\alpha_t} = \prod_{i=1}^{T} \alpha_i").scale(0.8)
        bar_alpha.next_to(to_x0, DOWN, buff=0.6)
        box = SurroundingRectangle(bar_alpha, color=BLUE, buff=0.2)
        self.play(Write(bar_alpha))
        # self.play(Create(box))
        self.wait(2)

        # Final form
        final = MathTex(
            r"x_t = \sqrt{\overline{\alpha_t}} x_0 + \sqrt{1-\overline{\alpha_t}} \epsilon_0"
        ).scale(0.85).next_to(bar_alpha, DOWN, buff=0.6)
        final_box = SurroundingRectangle(final, color=GREEN, buff=0.2)
        self.play(Write(final))
        self.play(Create(final_box))
        self.wait(2)

        self.play(FadeOut(final_box), FadeOut(pattern), FadeOut(dots), FadeOut(to_x0), FadeOut(bar_alpha))
        self.play(final.animate.next_to(section_title, DOWN, buff=1.0))
        
        # Equivalent distribution form
        dist_form = MathTex(
            r"x_t \sim \mathcal{N}(x_t; \sqrt{\overline{\alpha_t}}x_0, (1-\overline{\alpha_t})I)"
        ).scale(0.85).next_to(final, DOWN, buff=0.6)
        dist_box = SurroundingRectangle(dist_form, color=YELLOW, buff=0.2)
        self.play(Write(dist_form))
        self.play(Create(dist_box))
        self.wait(2)
        
        # Key insight
        insight = Text(
            "No need to sample q repeatedly!\nDirect sampling at any timestep t",
            font_size=28,
            color=ORANGE
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(insight))
        self.wait(3)
        
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )
        
        # Final summary
        summary_title = Text("Summary", font_size=40, weight=BOLD, color=GOLD)
        summary_points = VGroup(
            Text("✓ Reparameterization trick enables closed form", font_size=28),
            Text("✓ Sample directly at any timestep t", font_size=28),
            Text("✓ Much faster than sequential sampling", font_size=28),
            Text("✓ Key: Gaussian properties + recursive substitution", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        summary = VGroup(summary_title, summary_points).arrange(DOWN, buff=0.6)
        
        self.play(Write(summary_title))
        for point in summary_points:
            self.play(FadeIn(point, shift=RIGHT*0.3))
            self.wait(0.8)
        self.wait(3)


from manim import *
import numpy as np

class ReverseDiffusionVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Reverse Diffusion Process", font_size=48, color=BLUE)
        self.play(Write(title))
        self.wait()
        # self.play(title.animate.scale(0.5).to_edge(UP))
        self.play(FadeOut(title))
        
        # Part 1: Forward Process Recap
        self.forward_process_recap()
        self.wait(2)
        self.clear()
        
        # Part 2: The Reverse Challenge
        self.reverse_challenge()
        self.wait(2)
        self.clear()
        
        # Part 3: Neural Network Approximation
        self.neural_network_solution()
        self.wait(2)
        self.clear()
        
        # Part 4: Gaussian Assumption
        self.gaussian_assumption()
        self.wait(2)
        self.clear()
        
        # Part 5: Joint Distribution
        self.joint_distribution()
        self.wait(2)
        self.clear()
        
        # Part 6: Starting Point
        self.starting_point()
        self.wait(2)

    def forward_process_recap(self):
        """Show forward diffusion recap"""
        title = Text("Forward Diffusion: Adding Noise", font_size=36, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        
        ## Create image squares
        #num_steps = 5
        #squares = VGroup()
        #labels = VGroup()
        
        #for i in range(num_steps):
        #    # Create square with noise effect
        #    square = Square(side_length=1.2)
        #    noise_level = i / (num_steps - 1)
        #    square.set_fill(WHITE, opacity=1 - noise_level * 0.8)
        #    square.set_stroke(BLUE, width=2)
        #    
        #    # Add noise dots
        #    if i > 0:
        #        dots = VGroup(*[
        #            Dot(radius=0.02, color=RED).move_to(
        #                square.get_center() + 
        #                np.array([np.random.uniform(-0.5, 0.5), 
        #                         np.random.uniform(-0.5, 0.5), 0])
        #            ) for _ in range(int(20 * noise_level))
        #        ])
        #        square.add(dots)
        #    
        #    squares.add(square)
        #    
        #    # Add label
        #    if i == 0:
        #        label = MathTex(r"x_0", font_size=32)
        #        sublabel = Text("Clean Image", font_size=20, color=GREEN)
        #    elif i == num_steps - 1:
        #        label = MathTex(r"x_T", font_size=32)
        #        sublabel = Text("Pure Noise", font_size=20, color=RED)
        #    else:
        #        label = MathTex(f"x_{i}", font_size=32)
        #        sublabel = Text(f"Step {i}", font_size=20)
        #    
        #    label_group = VGroup(label, sublabel).arrange(DOWN, buff=0.1)
        #    labels.add(label_group)
        
        ## Arrange squares
        #squares.arrange(RIGHT, buff=0.5)
        #squares.move_to(ORIGIN)
        
        ## Position labels
        #for square, label in zip(squares, labels):
        #    label.next_to(square, DOWN, buff=0.3)
        
        ## Add arrows
        #arrows = VGroup()
        #for i in range(num_steps - 1):
        #    arrow = Arrow(
        #        squares[i].get_right(), 
        #        squares[i + 1].get_left(), 
        #        buff=0.1,
        #        color=YELLOW,
        #        stroke_width=3
        #    )
        #    arrows.add(arrow)
        
        ## Animate
        #self.play(
        #    *[Create(square) for square in squares],
        #    *[Write(label) for label in labels]
        #)
        #self.play(*[Create(arrow) for arrow in arrows])
        
        ## Show equation
        #eq = MathTex(
        #    r"x_t \text{ depends on } \beta_t \text{ and } x_{t-1}",
        #    font_size=32
        #)
        #eq.to_edge(DOWN)
        #self.play(Write(eq))

        # Scene 3: Forward Diffusion Process
        #title = Text("Forward Diffusion: Adding Noise Step by Step", 
        #            font_size=32).to_edge(UP)
        #self.play(Write(title))
        
        # Create circles representing states
        circles = VGroup()
        labels = VGroup()
        
        positions = [LEFT * 5, LEFT * 2, RIGHT * 1, RIGHT * 4]
        names = [r"x_T", r"x_t", r"x_{t-1}", r"x_0"]
        
        for pos, name in zip(positions, names):
            circle = Circle(radius=0.6, color=PINK, fill_opacity=0.3).move_to(pos)
            label = MathTex(name).move_to(circle)
            circles.add(circle)
            labels.add(label)
        
        # Create arrows with probability labels
        arrows = VGroup()
        prob_labels = VGroup()
        
        arrow_labels_text = [r"...", r"p_{\theta}(x_{t-1} | x_t)", r"..."]
        
        for i in range(len(circles) - 1):
            if i == 1:
                arrow = Arrow(circles[i].get_right(), circles[i+1].get_left(), 
                            buff=0.15, color=YELLOW)
            else:
                arrow = DashedLine(circles[i].get_right(), circles[i+1].get_left(), 
                                 buff=0.15).add_tip()
            arrows.add(arrow)
            
            if i == 1:
                prob_label = MathTex(arrow_labels_text[i], font_size=28)
                prob_label.next_to(arrow, UP, buff=0.1)
            elif i == 2:
                prob_label = MathTex(arrow_labels_text[i], font_size=28)
                prob_label.next_to(arrow, UP, buff=0.1)
            else:
                prob_label = MathTex(arrow_labels_text[i], font_size=28)
                prob_label.next_to(arrow, UP, buff=0.1)
            
            prob_labels.add(prob_label)
        
        # Add backward arrow from x_{t-1} to x_t
        #backward_arrow = Arrow(circles[2].get_left() + UP * 0.3, 
        #                      circles[1].get_left() + UP * 0.3, 
        #                      buff=0.15, color=GREEN)
        backward_arrow = Arrow(circles[2].get_left() + DOWN, 
                              circles[1].get_left() + DOWN, 
                              color=GREEN)
        backward_label = MathTex(r"q(x_t | x_{t-1})", font_size=28, color=GREEN)
        backward_label.next_to(backward_arrow, DOWN, buff=0.1)
        
        # Animate creation
        self.play(*[Create(c) for c in circles], 
                 *[Write(l) for l in labels])
        self.wait(1)
        
        self.play(*[Create(a) for a in arrows],
                 *[Write(p) for p in prob_labels])
        self.wait(1)
        
        self.play(Create(backward_arrow), Write(backward_label))

        # Create visual representation of data becoming noisy
        # Load an actual image from disk
        # Replace 'path/to/your/image.png' with your actual image path
        start_x = 4
        timesteps = len(circles)
        spacing = 2.0

        clean_image = ImageMobject(r"D:\youtube\manimations\diffusion\240_F_1434102712_XDZ4XBljlu4Ico4AE9HOXyGw3hmPsovt.jpg")  # Change this to your image path
        clean_image.scale(0.8)
        clean_image.move_to([start_x, -2.0, 0])

        # Create noisy version by overlaying random pixels
        noisy_dots = VGroup()
        np.random.seed(42)
        for _ in range(200):
            x = start_x - timesteps * spacing - 1 + np.random.uniform(-0.5, 0.5)
            y = -2.0 + np.random.uniform(-0.5, 0.5)
            dot = Dot(
                point=[x, y, 0],
                radius=0.015,
                color=random_bright_color()
            )
            noisy_dots.add(dot)
        
        # Show data transformation
        self.play(
            FadeIn(clean_image, scale=0.8),
            run_time=1.5
        )
        
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(dot, scale=0.5) for dot in noisy_dots],
                lag_ratio=0.005
            ),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Add noise visualization
        noise_text = Text("Pure Noise", font_size=20).next_to(circles[0], DOWN)
        clean_text = Text("Clean Image", font_size=20).next_to(circles[3], DOWN)
        
        self.play(FadeIn(noise_text), FadeIn(clean_text))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def reverse_challenge(self):
        """Show the reverse diffusion challenge"""
        title = Text("The Reverse Process: Can We Go Back?", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create noise and image
        noise_square = Square(side_length=2, color=RED, fill_opacity=0.3)
        noise_dots = VGroup(*[
            Dot(radius=0.03, color=RED).move_to(
                noise_square.get_center() + 
                np.array([np.random.uniform(-0.9, 0.9), 
                         np.random.uniform(-0.9, 0.9), 0])
            ) for _ in range(100)
        ])
        noise_group = VGroup(noise_square, noise_dots)
        noise_label = MathTex(r"x_T", font_size=36)
        noise_label.next_to(noise_square, DOWN)
        
        image_square = Square(side_length=2, color=GREEN, fill_opacity=0.5)
        image_label = MathTex(r"x_0", font_size=36)
        image_label.next_to(image_square, DOWN)
        
        noise_group.shift(LEFT * 3.5)
        noise_label.shift(LEFT * 3.5)
        image_square.shift(RIGHT * 3.5)
        image_label.shift(RIGHT * 3.5)
        
        # Question mark
        question = Text("?", font_size=80, color=YELLOW)
        question.move_to(ORIGIN)
        
        # Arrow
        arrow = Arrow(noise_group.get_right(), image_square.get_left(), 
                     buff=0.5, color=BLUE, stroke_width=6)
        
        self.play(
            Create(noise_group),
            Write(noise_label),
            Create(image_square),
            Write(image_label)
        )
        self.play(Create(arrow), Write(question))
        
        # Show the probability equation
        eq1 = MathTex(r"p(x_{t-1} | x_t) = \, ?", font_size=40)
        eq1.move_to(ORIGIN + UP * 0.5)
        
        self.play(Transform(question, eq1))
        self.wait()
        
        # Show the problem
        problem = Text("Intractable: Requires knowing ALL possible images!", 
                      font_size=28, color=RED)
        problem.to_edge(DOWN)
        self.play(Write(problem))

    def neural_network_solution(self):
        """Show neural network approximation"""
        title = Text("Neural Network Approximation", 
                    font_size=36, color=GREEN)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create neural network diagram
        network = self.create_simple_network()
        network.scale(0.8).move_to(ORIGIN + LEFT * 3)
        
        # Input and output
        input_label = MathTex(r"x_t, t", font_size=32)
        input_label.next_to(network, LEFT, buff=0.5)
        
        output_label = MathTex(r"p_\theta(x_{t-1}|x_t)", font_size=32)
        output_label.next_to(network, RIGHT, buff=0.5)
        
        self.play(Create(network))
        self.play(Write(input_label), Write(output_label))
        
        # Show parameterization
        # eq_box = Rectangle(width=6, height=2.5, color=BLUE)
        # eq_box.move_to(RIGHT * 3.5 + DOWN * 0.5)
        
        eq1 = MathTex(r"p_\theta(x_{t-1}|x_t) = ", font_size=45)
        eq2 = MathTex(r"\mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))", 
                     font_size=45)
        
        equations = VGroup(eq1, eq2).arrange(DOWN, buff=0.3)
        # equations.move_to(eq_box.get_center())
        equations.move_to(RIGHT * 3.5 + UP)
        eq_box = SurroundingRectangle(equations, color=YELLOW)
        
        self.play(Create(eq_box))
        self.play(Write(equations))
        
        # Key points
        points = VGroup(
            Text("• θ = neural network parameters", font_size=22),
            Text("• Learn during training", font_size=22),
            Text("• Approximates true reverse process", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        points.next_to(eq_box, DOWN, buff=0.3)
        
        self.play(Write(points))

    def create_simple_network(self):
        """Create a simple neural network diagram"""
        layers = VGroup()
        
        # Input layer
        input_layer = VGroup(*[Circle(radius=0.15, color=BLUE, fill_opacity=0.7) 
                              for _ in range(3)])
        input_layer.arrange(DOWN, buff=0.2)
        
        # Hidden layer
        hidden_layer = VGroup(*[Circle(radius=0.15, color=GREEN, fill_opacity=0.7) 
                               for _ in range(4)])
        hidden_layer.arrange(DOWN, buff=0.2)
        hidden_layer.shift(RIGHT * 1.5)
        
        # Output layer
        output_layer = VGroup(*[Circle(radius=0.15, color=RED, fill_opacity=0.7) 
                               for _ in range(3)])
        output_layer.arrange(DOWN, buff=0.2)
        output_layer.shift(RIGHT * 3)
        
        # Connections
        connections = VGroup()
        for i in input_layer:
            for h in hidden_layer:
                connections.add(Line(i.get_center(), h.get_center(), 
                                   stroke_width=1, color=GRAY))
        
        for h in hidden_layer:
            for o in output_layer:
                connections.add(Line(h.get_center(), o.get_center(), 
                                   stroke_width=1, color=GRAY))
        
        network = VGroup(connections, input_layer, hidden_layer, output_layer)
        return network

    def gaussian_assumption(self):
        """Show Gaussian assumption with Brownian motion simulation"""
        title = Text("Brownian Motion & Gaussian Property", 
                    font_size=36, color=PURPLE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create main axes for Brownian paths
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[-30, 30, 10],
            x_length=9,
            y_length=5,
            axis_config={"color": BLUE, "include_tip": False},
        )
        axes.shift(LEFT * 1.5)
        
        labels = axes.get_axis_labels(
            x_label=MathTex("t", font_size=32),
            y_label=MathTex("X(t)", font_size=32)
        )
        
        self.play(Create(axes), Write(labels))
        
        # Simulate multiple Brownian motion paths
        num_paths = 30
        num_steps = 100
        dt = 1.0
        
        paths_group = VGroup()
        
        # Generate and animate paths
        for path_idx in range(num_paths):
            # Generate Brownian motion path
            increments = np.random.normal(0, np.sqrt(dt), num_steps)
            path = np.cumsum(increments)
            path = np.insert(path, 0, 0)  # Start at 0
            
            # Scale the path for better visualization
            # path = path * 3
            
            # Create path points
            points = [axes.c2p(t, path[t]) for t in range(len(path))]
            
            # Color gradient from blue to red based on path index
            color_value = path_idx / num_paths
            if color_value < 0.5:
                path_color = interpolate_color(BLUE, YELLOW, color_value * 2)
            else:
                path_color = interpolate_color(YELLOW, RED, (color_value - 0.5) * 2)
            
            # Create the path
            path_line = VMobject(stroke_width=1.5, color=path_color)
            path_line.set_points_smoothly(points)
            path_line.set_opacity(0.6)
            
            paths_group.add(path_line)
        
        # Animate paths appearing
        self.play(
            *[Create(path) for path in paths_group],
            run_time=3,
            lag_ratio=0.02
        )
        
        # Add expected value line
        expected_line = axes.plot(
            lambda t: 0,
            color=BLUE,
            stroke_width=4
        )
        expected_label = MathTex(r"E[X_t]", font_size=28, color=BLUE)
        expected_label.next_to(axes.c2p(10, 0), UP, buff=0.2)
        
        self.play(
            Create(expected_line),
            Write(expected_label)
        )
        
        # Create histogram on the right side for X_T distribution
        hist_axes = Axes(
            x_range=[0, 0.04, 0.01],
            y_range=[-30, 30, 10],
            x_length=2.5,
            y_length=5,
            axis_config={"color": YELLOW, "include_tip": False},
        )
        hist_axes.shift(RIGHT * 4.5)
        
        hist_x_label = MathTex(r"X_T", font_size=24, color=YELLOW)
        hist_x_label.next_to(hist_axes, DOWN, buff=0.2)
        
        self.play(Create(hist_axes), Write(hist_x_label))
        
        # Collect final values from all paths
        final_values = []
        for path in paths_group:
            final_point = path.points[-1]
            final_y = axes.p2c(final_point)[1]
            final_values.append(final_y)
        
        # Create histogram bars
        bins = np.linspace(-30, 30, 15)
        hist, bin_edges = np.histogram(final_values, bins=bins, density=True)
        
        bars = VGroup()
        for i in range(len(hist)):
            bar_height = hist[i]
            bar_bottom = bin_edges[i]
            bar_top = bin_edges[i + 1]
            bar_center = (bar_bottom + bar_top) / 2
            
            # Color gradient for histogram
            color_val = (i / len(hist))
            if color_val < 0.5:
                bar_color = interpolate_color(BLUE, YELLOW, color_val * 2)
            else:
                bar_color = interpolate_color(YELLOW, RED, (color_val - 0.5) * 2)
            
            bar = Rectangle(
                width=bar_height * 50,  # Scale for visibility
                height=(bar_top - bar_bottom) * 0.9,
                fill_color=bar_color,
                fill_opacity=0.7,
                stroke_width=1,
                stroke_color=WHITE
            )
            bar.move_to(hist_axes.c2p(bar_height / 2, bar_center))
            bars.add(bar)
        
        self.play(*[GrowFromEdge(bar, LEFT) for bar in bars], run_time=2)
        
        # Overlay theoretical Gaussian curve
        gaussian_curve = hist_axes.plot(
            lambda x: (1/np.sqrt(2*np.pi*100)) * np.exp(-0.5*(x**2)/100),
            color=YELLOW,
            stroke_width=3,
            x_range=[0, 0.04]
        )
        
        self.play(Create(gaussian_curve))
        
        # Add subtitle
        subtitle = Text(
            "Simulated Paths Xt, t ∈ [t₀, T]",
            font_size=24,
            color=WHITE
        )
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(subtitle))
        
        # Key insight box
        insight_box = Rectangle(
            width=5,
            height=1.8,
            color=GREEN,
            fill_opacity=0.1,
            stroke_width=2
        )
        insight_box.to_corner(3*DOWN, buff=0.3)
        
        insight = VGroup(
            Text("Key Insight:", font_size=22, color=GREEN, weight=BOLD),
            Text("• Small steps → Gaussian", font_size=18),
            Text("• X_T distribution is Normal", font_size=18),
            Text("• Validates reverse process", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        insight.move_to(insight_box.get_center())
        
        self.play(Create(insight_box), Write(insight))

    def joint_distribution(self):
        """Show joint distribution factorization"""
        title = Text("Joint Distribution: Product of Steps", 
                    font_size=36, color=ORANGE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Main equation
        eq1 = MathTex(
            r"p_\theta(x_1, ..., x_T) = p_\theta(x_{0:T})",
            font_size=45
        )
        eq1.move_to(UP * 2)
        
        self.play(Write(eq1))
        self.wait()
        
        # Factorization
        eq2 = MathTex(
            r"= p_\theta(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)",
            font_size=45
        )
        eq2.next_to(eq1, DOWN, buff=0.5)
        
        self.play(Write(eq2))
        
        # Visual representation of product
        circles = VGroup()
        for i in range(5):
            if i == 0:
                circle = Circle(radius=0.4, color=RED, fill_opacity=0.3)
                label = MathTex(r"p(x_T)", font_size=20)
            else:
                circle = Circle(radius=0.4, color=GREEN, fill_opacity=0.3)
                label = MathTex(f"p(x_{{{4-i}}}|x_{{{5-i}}})", font_size=18)
            
            label.move_to(circle.get_center())
            group = VGroup(circle, label)
            circles.add(group)
        
        circles.arrange(RIGHT, buff=0.3)
        circles.move_to(DOWN * 1)
        
        # Multiplication symbols
        mult_symbols = VGroup()
        for i in range(4):
            mult = MathTex(r"\times", font_size=32)
            mult.move_to(
                (circles[i].get_right() + circles[i+1].get_left()) / 2
            )
            mult_symbols.add(mult)
        
        self.play(
            *[Create(circle) for circle in circles],
            *[Create(mult) for mult in mult_symbols]
        )
        
        # Explanation
        explanation = Text(
            "Chain rule: Break down into manageable steps",
            font_size=26,
            color=YELLOW
        )
        explanation.to_edge(DOWN)
        self.play(Write(explanation))

    def starting_point(self):
        """Show the starting point - pure Gaussian noise"""
        title = Text("Starting Point: Pure Gaussian Noise", 
                    font_size=36, color=RED)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Main equation
        eq = MathTex(
            r"p_\theta(x_T) = \mathcal{N}(x_T; 0, I)",
            font_size=48
        )
        eq.move_to(UP * 1.5)
        self.play(Write(eq))
        
        # Create visual representation
        # Standard normal distribution
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 0.5, 0.1],
            x_length=6,
            y_length=3,
            axis_config={"color": BLUE},
        )
        axes.move_to(LEFT * 3)
        
        gaussian = axes.plot(
            lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2),
            color=RED,
            stroke_width=3
        )
        
        graph_label = Text("Standard Normal\nμ=0, σ=1", font_size=20)
        graph_label.next_to(axes, DOWN)
        
        self.play(Create(axes), Create(gaussian), Write(graph_label))
        
        # Noise visualization
        noise_square = Square(side_length=3, color=RED)
        noise_square.move_to(RIGHT * 3 + DOWN * 1.5)
        
        noise_dots = VGroup(*[
            Dot(radius=0.02, color=RED).move_to(
                noise_square.get_center() + 
                np.array([np.random.normal(0, 0.8), 
                         np.random.normal(0, 0.8), 0])
            ) for _ in range(200)
        ])
        
        noise_label = MathTex(r"x_T", font_size=32)
        noise_label.next_to(noise_square, DOWN)
        
        self.play(
            Create(noise_square),
            *[Create(dot) for dot in noise_dots],
            Write(noise_label)
        )
        
        # Summary points
        summary = VGroup(
            Text("• Mean = 0 (centered)", font_size=24, color=GREEN),
            Text("• Variance = I (unit variance)", font_size=24, color=GREEN),
            Text("• This is our starting point for generation!", 
                font_size=24, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        summary.to_edge(DOWN, buff=0.3)
        
        self.play(Write(summary))
        self.wait(2)
        
        # Final message
        final = Text(
            "From this noise, we reverse diffuse to create images!",
            font_size=32,
            color=GREEN
        )
        final.move_to(ORIGIN)
        
        self.play(
            FadeOut(axes), FadeOut(gaussian), FadeOut(graph_label),
            FadeOut(noise_square), FadeOut(noise_dots), FadeOut(noise_label),
            FadeOut(summary), FadeOut(eq)
        )
        self.play(Write(final))
        self.wait(2)


from manim import *

class ELBOVisualization(Scene):
    def construct(self):
        self.brown = "#8B4513"

        self.introduction_and_problem()
        self.wait(1)
        self.approximate_posterior()
        self.wait(1)
        eq31 = self.deriving_prior()
        self.wait(1)
        eq34 = self.log_probability_derivation(eq31)
        self.wait(1)
        eq36 = self.introduce_approximation(eq34)
        self.wait(1)
        eq43 = self.splitting_the_expectation(eq36)
        self.wait(1)
        self.the_elbo(eq43)
        self.wait(1)
        self.lever_analogy()
        self.wait(1)
        self.maximizing_elbo()
        self.wait(1)

    
    def introduction_and_problem(self):
        """Introduction and the intractability problem"""
        title = Text("Evidence Lower Bound (ELBO)", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Setup
        setup = VGroup(
            MathTex(r"\text{Data: } x, \quad \text{Latent: } z", font_size=32),
            MathTex(r"\text{Goal: Learn } p(x)", color=YELLOW, font_size=36)
        ).arrange(DOWN, buff=0.4)
        setup.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(setup))
        self.wait(1.5)
        
        # The problem
        problem_title = Text("The Problem:", font_size=32, color=RED, weight=BOLD)
        problem_title.next_to(setup, DOWN, buff=0.8)
        
        self.play(FadeIn(problem_title))
        self.wait(0.5)
        
        # Two formulations side by side
        eq_marginal = MathTex(
            r"p(x) = \int p(x, z) dz",
            font_size=38
        )
        
        eq_bayes = MathTex(
            r"p(x) = \frac{p(x, z)}{p(z|x)}",
            font_size=38
        )
        
        equations = VGroup(eq_marginal, eq_bayes).arrange(RIGHT, buff=1.5)
        equations.next_to(problem_title, DOWN, buff=0.5)
        
        # Labels below each equation
        label_marginal = Text("Intractable integral", font_size=24, color=RED)
        label_marginal.next_to(eq_marginal, DOWN, buff=0.3)
        
        label_bayes = Text("Need true p(z|x)", font_size=24, color=ORANGE)
        label_bayes.next_to(eq_bayes, DOWN, buff=0.3)
        
        self.play(Write(eq_marginal), Write(eq_bayes))
        self.wait(1)
        self.play(FadeIn(label_marginal), FadeIn(label_bayes))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, setup, problem_title, equations, label_marginal, label_bayes)))
   
    def approximate_posterior(self):
        """Solution: Approximate the Posterior"""
        title = Text("Solution: Approximate the Posterior", font_size=40, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        approx = VGroup(
            Text("Instead of true posterior p(z|x):", font_size=32),
            MathTex(r"\text{Learn approximate posterior: } q_\phi(z|x)", font_size=36, color=GREEN),
            Text("where φ are learnable parameters", font_size=28, color=GRAY)
        ).arrange(DOWN, buff=0.4)
        approx.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(approx[0]))
        self.wait(0.5)
        self.play(Write(approx[1]))
        self.wait(0.5)
        self.play(FadeIn(approx[2]))
        self.wait(2)
        
        eq27_box = Rectangle(width=10, height=1.5, color=YELLOW)
        eq27_box.next_to(approx, DOWN, buff=0.8)
        
        eq27 = MathTex(
            r"q_\phi(z, x) = q_\phi(z|x) q_\phi(x)",
            font_size=44
        ).move_to(eq27_box)
        
        self.play(Create(eq27_box))
        self.play(Write(eq27))
        self.wait(2)
        
        explanation = Text(
            "Joint distribution factorizes into conditional and marginal",
            font_size=26,
            color=GRAY
        )
        explanation.next_to(eq27_box, DOWN, buff=0.3)
        self.play(FadeIn(explanation))
        self.wait(3)
        
        self.play(FadeOut(VGroup(title, approx, eq27_box, eq27, explanation)))
    
    def deriving_prior(self):
        """Deriving the Prior q(x)"""
        title = Text("Deriving the Prior q(x)", font_size=40, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        eq28 = MathTex(
            r"q_\phi(x) = \int q_\phi(z, x) dz",
            font_size=100
        )
        
        self.play(Write(eq28))
        self.wait(2)
        
        eq29 = MathTex(
            r"q_\phi(x) = \int q_\phi(z|x) q_\phi(x) dz",
            font_size=100
        )

        arrow1 = Arrow(start=DOWN, end=UP, color=GREEN).scale(0.5)
        arrow1.next_to(eq28, DOWN, buff=0.3)
        
        sub_text = Text("Substitute factorization", font_size=26, color=GREEN)
        sub_text.next_to(arrow1, DOWN, buff=0.2)
        
        self.play(GrowArrow(arrow1), FadeIn(sub_text))
        self.wait(1)

        self.play(ReplacementTransform(eq28, eq29))
        self.wait(2)

        self.play(FadeOut(arrow1, sub_text))
        self.wait(1)
        
        arrow2 = Arrow(start=3*DOWN + 3*RIGHT, end=DOWN + 3*RIGHT, color=ORANGE).scale(0.5)
        # arrow2.next_to(eq29, DOWN + 0.5*RIGHT, buff=0.3)
        
        factor_text = Text("Factor out q(x)", font_size=26, color=ORANGE)
        factor_text.next_to(arrow2, DOWN, buff=0.2)
        
        self.play(GrowArrow(arrow2), FadeIn(factor_text))
        self.wait(1)
        
        eq30 = MathTex(
            r"\Rightarrow q_\phi(x) = q_\phi(x) \int q_\phi(z|x) dz",
            font_size=100
        )
        # eq30.next_to(factor_text, DOWN, buff=0.5)
        
        # self.play(Write(eq30))
        self.play(ReplacementTransform(eq29, eq30))
        self.wait(2)

        self.play(FadeOut(arrow2, factor_text))
        self.wait(1)

        eq31 = MathTex(
            r"1 = \int q_\phi(z|x) dz",
            font_size=120,
            color=GREEN
        )

        self.play(ReplacementTransform(eq30, eq31))
        self.wait(2)
        
        self.play(FadeOut(title))
        self.play(eq31.animate.set_color(WHITE))
        return eq31
    
    def log_probability_derivation(self, eq31):
        """Taking the Log"""
        title = Text("Taking the Log", font_size=40, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        #eq31 = MathTex(
        #    r"\int q_\phi(z|x) dz = 1",
        #    font_size=100
        #).shift(UP*2)
        
        #self.play(Write(eq31))
        #self.wait(1)
        
        step1_text = Text("Multiply both sides by log p(x)", font_size=28, color=GREEN)
        step1_text.next_to(eq31, DOWN, buff=0.5)
        
        self.play(FadeIn(step1_text))
        self.wait(1)
        
        eq32 = MathTex(
            r"\Rightarrow \log p(x) = \log p(x) \int q_\phi(z|x) dz",
            font_size=80
        )
        # eq32.next_to(step1_text, DOWN, buff=0.4)
        
        # self.play(Write(eq32))
        self.play(ReplacementTransform(eq31, eq32))
        self.wait(2)

        self.play(FadeOut(step1_text))
        self.wait(1)
        
        step2_text = Text("Move log p(x) inside the integral", font_size=28, color=ORANGE)
        step2_text.next_to(eq32, DOWN, buff=0.5)
        
        self.play(FadeIn(step2_text))
        self.wait(1)
        
        eq33 = MathTex(
            r"\log p(x) = \int q_\phi(z|x) \log p(x) dz",
            font_size=100
        )
        # eq33.next_to(step2_text, DOWN, buff=0.4)
        
        # self.play(Write(eq33))
        self.play(ReplacementTransform(eq32, eq33))
        self.wait(2)

        self.play(FadeOut(step2_text))
        self.wait(1)
        
        step3_text = Text("Rewrite as expectation", font_size=28, color=PURPLE)
        step3_text.next_to(eq33, DOWN, buff=0.5)
        
        self.play(FadeIn(step3_text))
        self.wait(1)
        
        eq34 = MathTex(
            r"\log p(x) = \mathbb{E}_{q_\phi(z|x)}[\log p(x)]",
            font_size=100
        )
        # eq34.next_to(step3_text, DOWN, buff=0.4)
        
        # self.play(Write(eq34))
        self.play(ReplacementTransform(eq33, eq34))
        self.wait(3)
        
        # self.play(FadeOut(VGroup(title, eq31, step1_text, eq32, step2_text, eq33, step3_text, eq34)))
        self.play(FadeOut(title, step3_text))

        return eq34
    
    def introduce_approximation(self, eq34):
        """Introducing the Approximation"""
        title = Text("Introducing the Approximation", font_size=40, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # recall = Text("Recall:", font_size=28, color=GRAY)
        # recall.next_to(title, DOWN, buff=0.5)
        
        eq26 = MathTex(
            r"p(x) = \frac{p(x, z)}{p(z|x)}",
            font_size=45
        ).next_to(eq34, DOWN, buff=0.5)
        
        # self.play(FadeIn(recall), Write(eq26))
        self.play(Write(eq26))
        self.wait(1)
        
        #insight = Text(
        #    "Key Insight: Introduce our approximation q(z|x)",
        #    font_size=30,
        #    color=YELLOW
        #)
        #insight.next_to(eq26, DOWN, buff=0.6)
        
        # self.play(Write(insight))
        # self.wait(2)
        
        eq35 = MathTex(
            r"\log p(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{p(z|x)}\right]",
            font_size=80
        )
        # eq35.next_to(insight, DOWN, buff=0.6)
        
        # self.play(Write(eq35))
        self.play(ReplacementTransform(eq34, eq35))
        self.wait(2)

        self.play(FadeOut(eq26))
        self.wait(1)
        
        trick = Text(
            "Multiply numerator & denominator by q(z|x)",
            font_size=28,
            color=GREEN
        )
        trick.next_to(eq35, DOWN, buff=0.5)
        
        self.play(FadeIn(trick))
        self.wait(2)
        
        # self.play(FadeOut(VGroup(title, recall, eq26, insight, eq35, trick)))

        eq36 = MathTex(
            r"\log p(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{p(z|x)} \cdot \frac{q_\phi(z|x)}{q_\phi(z|x)}\right]",
            font_size=70
        )#.shift(UP*2)
        
        # self.play(Write(eq36))
        self.play(ReplacementTransform(eq35, eq36))
        self.wait(2)

        self.play(FadeOut(title, trick))

        return eq36
    
    def splitting_the_expectation(self, eq36):
        """Splitting the Terms"""
        title = Text("Splitting the Terms", font_size=40, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        prop_text = Text(
            "Property: log(AB) = log A + log B",
            font_size=26,
            color=GREEN
        )
        prop_text.next_to(eq36, DOWN, buff=0.4)
        
        self.play(FadeIn(prop_text))
        self.wait(1.5)
        
        eq42 = MathTex(
            r"\log p(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{q_\phi(z|x)} + \log \frac{q_\phi(z|x)}{p(z|x)}\right]",
            font_size=60
        )
        # eq42.next_to(prop_text, DOWN, buff=0.5)
        
        # self.play(Write(eq42))
        self.play(ReplacementTransform(eq36, eq42))
        self.wait(2)

        self.play(FadeOut(prop_text))
        self.wait(1)
        
        linearity = Text(
            "E[X + Y] = E[X] + E[Y]  (Linearity of Expectation)",
            font_size=26,
            color=ORANGE
        )
        linearity.next_to(eq42, DOWN, buff=0.4)
        
        self.play(FadeIn(linearity))
        self.wait(1.5)
        
        eq43 = MathTex(
            r"\log p(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{q_\phi(z|x)}\right] + \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p(z|x)}\right]",
            font_size=50
        )
        # eq43.next_to(linearity, DOWN, buff=0.5)
        
        # self.play(Write(eq43))
        self.play(ReplacementTransform(eq42, eq43))
        self.wait(2)
        
        # self.play(FadeOut(VGroup(title, eq36, prop_text, eq42, linearity, eq43)))
        self.play(FadeOut(title, linearity))

        return eq43
    
    def the_elbo(self, eq43):
        """The ELBO Emerges!"""
        title = Text("The ELBO Emerges!", font_size=44, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # eq44_box = Rectangle(width=13, height=2.5, color=YELLOW, stroke_width=3)
        # eq44_box.shift(UP*0.5)
        
        eq44 = MathTex(
            r"\log p(x) &= \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{q_\phi(z|x)}\right] + D_{KL}(q_\phi(z|x) || p(z|x))",
            font_size=50
        )#.move_to(eq44_box)
        
        # self.play(Create(eq44_box))
        # self.play(Write(eq44))
        self.play(ReplacementTransform(eq43, eq44))
        self.wait(2)

        elbo = MathTex(
            r"ELBO = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{q_\phi(z|x)}\right]",
            font_size=30
        ).next_to(eq44, DOWN, buff=0.4)
        
        # brace1 = Brace(eq44, DOWN, color=GREEN)
        # text1 = Text("ELBO", font_size=28, color=GREEN).next_to(brace1, DOWN + LEFT)

        kl = MathTex(
            r"\textbf{KL div} = D_{KL}(q_\phi(z|x) || p(z|x)) \ge 0",
            font_size=30
        ).next_to(elbo, DOWN, buff=0.4)
        
        # brace2 = Brace(eq44, DOWN, color=RED)
        # text2 = Text("KL Divergence ≥ 0", font_size=28, color=RED).next_to(brace2, DOWN + RIGHT)
        
        # self.play(GrowFromCenter(brace1), FadeIn(text1))
        self.play(Write(elbo))
        self.wait(1)
        # self.play(GrowFromCenter(brace2), FadeIn(text2))
        # self.wait(2)
        self.play(Write(kl))
        self.wait(1)
        
        # arrow = Arrow(LEFT, RIGHT, color=ORANGE).scale(0.7)
        # arrow.next_to(eq44, DOWN, buff=0.8)
        
        # conclusion_text = Text("Since KL ≥ 0:", font_size=28, color=ORANGE)
        # conclusion_text.next_to(arrow, LEFT, buff=0.3)
        
        # self.play(GrowArrow(arrow), FadeIn(conclusion_text))
        # self.wait(1)

        self.play(FadeOut(VGroup(eq44, elbo, kl)))
        
        eq45_box = Rectangle(width=11, height=1.8, color=GREEN, stroke_width=4)
        eq45_box.next_to(title, DOWN, buff=0.5)
        
        eq45 = MathTex(
            r"\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{q_\phi(z|x)}\right]",
            font_size=60,
            color=GREEN
        ).move_to(eq45_box)
        
        elbo_label = Text(
            "Evidence Lower Bound (ELBO)",
            font_size=32,
            color=GREEN,
            weight=BOLD
        )
        elbo_label.next_to(eq45_box, DOWN, buff=0.4)
        
        self.play(Create(eq45_box))
        self.play(Write(eq45))
        self.wait(1)
        self.play(Write(elbo_label))
        self.wait(3)
        
        self.play(FadeOut(VGroup(title, eq45_box, eq45, elbo_label)))
    
    def lever_analogy(self):
        """The Lever Analogy"""
        title = Text("The Lever Analogy", font_size=44, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        explanation = VGroup(
            Text("Why is it called a 'Lower Bound'?", font_size=32),
            Text("It's difficult to lift the original object directly", font_size=28, color=GRAY),
            Text("(computing log p(x) is intractable)", font_size=26, color=GRAY, slant=ITALIC)
        ).arrange(DOWN, buff=0.3)
        explanation.next_to(title, DOWN, buff=0.5)
        
        self.play(FadeIn(explanation))
        self.wait(3)
        
        lever_group = VGroup()
        
        fulcrum = Triangle(color=GRAY, fill_opacity=1).scale(0.5)
        fulcrum.shift(DOWN*0.5)
        
        lever = Rectangle(width=8, height=0.2, color=self.brown, fill_opacity=1)
        lever.move_to(fulcrum.get_top() + UP*0.1)
        
        left_box = Rectangle(width=1.5, height=1, color=GREEN, fill_opacity=0.5)
        left_box.next_to(lever.get_left(), UP, buff=0)
        left_label = Text("ELBO\n(computable)", font_size=20, color=GREEN)
        left_label.move_to(left_box)
        
        right_box = Rectangle(width=1.5, height=1.5, color=RED, fill_opacity=0.5)
        right_box.next_to(lever.get_right(), UP, buff=0).shift(DOWN*0.3)
        right_label = Text("log p(x)\n(intractable)", font_size=20, color=RED)
        right_label.move_to(right_box)
        
        lever_group.add(fulcrum, lever, left_box, left_label, right_box, right_label)
        lever_group.scale(0.8).shift(DOWN*1)
        
        self.play(FadeOut(explanation))
        self.play(Create(lever_group))
        self.wait(2)
        
        inequality = MathTex(
            r"\text{ELBO} \leq \log p(x)",
            font_size=36,
            color=YELLOW
        )
        inequality.next_to(lever_group, DOWN, buff=0.8)
        
        self.play(Write(inequality))
        self.wait(3)
        
        self.play(FadeOut(VGroup(title, lever_group, inequality)))
    
    def maximizing_elbo(self):
        """Maximizing the ELBO"""
        title = Text("Maximizing the ELBO", font_size=44, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        strategy = VGroup(
            Text("Training Strategy:", font_size=32, weight=BOLD),
            Text("• Maximize ELBO instead of log p(x)", font_size=28),
            Text("• This indirectly maximizes log p(x)", font_size=28),
            Text("• Equivalent to minimizing KL divergence", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        strategy.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(strategy[0]))
        for i in range(1, len(strategy)):
            self.play(FadeIn(strategy[i]))
            self.wait(1)
        
        self.wait(2)
        
        relationship = MathTex(
            r"\text{Maximize: } \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{q_\phi(z|x)}\right]",
            font_size=32,
            color=GREEN
        )
        relationship.next_to(strategy, DOWN, buff=0.8)
        
        relationship_box = SurroundingRectangle(relationship, color=GREEN, buff=0.3)
        
        self.play(Create(relationship_box), Write(relationship))
        self.wait(2)
        
        benefits = VGroup(
            Text("Benefits:", font_size=32, weight=BOLD, color=YELLOW),
            Text("✓ Tractable to compute", font_size=26),
            Text("✓ Can be optimized with gradient descent", font_size=26),
            Text("✓ Learns good approximation q(z|x)", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        benefits.next_to(relationship_box, DOWN, buff=0.8)
        
        self.play(FadeIn(benefits[0]))
        for i in range(1, len(benefits)):
            self.play(FadeIn(benefits[i]))
            self.wait(0.8)
        
        self.wait(3)
        
        self.play(FadeOut(VGroup(title, strategy, relationship_box, relationship, benefits)))


from manim import *

class DDPMELBODecomposition(Scene):
    def construct(self):
        self.camera.background_color = "#0f0f23"
        
        # Title
        self.show_title()
        self.wait(1)
        
        # Step 1: Starting ELBO
        self.show_starting_elbo()
        
        # Step 2: Jensen's inequality
        eq5 = self.show_jensens_inequality()
        
        # Step 4: Expand reverse process
        reverse_expansion_eq = self.show_reverse_expansion(eq5)
        
        ## Step 5: Add x0 conditioning
        x0_conditioning_eq = self.show_x0_conditioning(reverse_expansion_eq)
        
        # Step 6: Apply Bayes rule
        # self.show_bayes_rule(x0_conditioning_eq)
        
        # Step 7: Substitute Bayes
        eq57 = self.show_bayes_substitution(x0_conditioning_eq)

        # Step 8: Focus on E2
        eq_e2 = self.show_focus_on_e2(eq57)

        # Step 9: Apply Bayes to E2
        eq_e2 = self.show_bayes_on_e2(eq_e2)

        # Step 10: Separate terms
        e2_term_separation = self.show_term_separation(eq_e2)

        # Step 11: Cancellation animation
        self.show_cancellation_animation(e2_term_separation)
        
        # Step 9: Cancellation
        self.show_cancellation(e2_term_separation)
        
        # Step 10: Three component form
        self.show_three_components()
        
        # Step 11: Apply linearity
        linear_eq = self.show_linearity()
        
        # Step 12: Final decomposition
        self.show_final_decomposition(linear_eq)

        # fade out all the objects
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        
        # Step 13: Explain L0
        self.explain_l0()
        
        ## Step 14: Explain LT
        self.explain_lt()
        
        # Step 15: Explain sum terms
        self.explain_sum_terms()
        
        # Step 16: Why it matters
        self.show_why_matters()
        
        # Step 17: Final summary
        self.show_final_summary()

    def show_title(self):
        title = Text("DDPM ELBO Decomposition", font_size=48, color=BLUE)
        subtitle = Text("Making the Loss Tractable", font_size=32, color=PURPLE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

    def show_starting_elbo(self):
        desc = Text("DDPM ELBO Decomposition", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        eq = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right]",
            font_size=80
        )
        
        self.play(Write(desc))
        self.play(Write(eq))
        self.wait(3)
        # self.play(FadeOut(desc), FadeOut(eq))
        self.play(FadeOut(eq))

    def show_jensens_inequality(self):
        eq1 = MathTex(
            r"\log p(x) = \log \int p(x_{0:T})dx_{1:T}",
            font_size=80
        )
        
        eq2 = MathTex(
            r"\log p(x) = \log \int \frac{p(x_{0:T})q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} dx_{1:T}",
            font_size=70
        )
        
        eq3 = MathTex(
            r"\log p(x) = \log \mathbb{E}_{q(x_{1:T}|x_0)}\left[\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right]",
            font_size=70
        )
        
        eq4 = MathTex(
            r"\log p(x) \geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right]",
            font_size=70
        )

        eq5 = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right]",
            font_size=70
        )
        
        self.play(Write(eq1))
        self.wait(1)
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(1)
        self.play(ReplacementTransform(eq2, eq3))
        self.wait(1)
        self.play(ReplacementTransform(eq3, eq4))
        self.wait(3)
        self.play(ReplacementTransform(eq4, eq5))

        # self.play(Write(eq4))
        self.wait(2)
        # self.play(FadeOut(desc), FadeOut(eq1), FadeOut(eq2), FadeOut(eq3), FadeOut(eq4))
        # self.play(FadeOut(eq5))
        return eq5

    def show_reverse_expansion(self, eq5):
        eq = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)}{q(x_{1:T}|x_0)}\right]",
            font_size=60
        )
        
        self.play(ReplacementTransform(eq5, eq))
        self.wait(2)
        return eq

        # self.play(FadeOut(eq))

    def show_x0_conditioning(self, reverse_expansion_eq):
        desc = Text("Adding x₀ Conditioning (Markovian)", font_size=36, color=GREEN)
        desc.next_to(reverse_expansion_eq, DOWN, buff=1.0)
        # self.play(Write(desc))
        # desc = Text("Adding x₀ Conditioning (Markovian)", font_size=36, color=YELLOW)
        # desc.to_edge(UP)
        
        x0_conditioning_eq = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)p_\theta(x_0|x_1)\prod_{t=2}^T p_\theta(x_{t-1}|x_t)}{q(x_1|x_0)\prod_{t=2}^T q(x_t|x_{t-1},x_0)}\right]",
            font_size=50
        )
        
        self.play(Write(desc))
        self.wait(2)
        # self.play(Write(x0_conditioning_eq))
        self.play(ReplacementTransform(reverse_expansion_eq, x0_conditioning_eq))
        self.wait(3)

        self.play(FadeOut(desc))
        self.wait(2)

        return x0_conditioning_eq

    def show_bayes_rule(self, x0_conditioning_eq):
        eq2 = MathTex(
            r"= \frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}",
            font_size=44
        )

        self.play(x0_conditioning_eq.animate.shift(2 * UP).scale(0.5))
        self.wait(1)

        desc = Text("x0 conditioning", font_size=30, color=GREEN)
        desc.next_to(x0_conditioning_eq, DOWN, buff=0.5)

        eq1 = MathTex(
            r"q(x_t|x_{t-1}) = q(x_t|x_{t-1},x_0)",
            font_size=44
        ).next_to(desc, DOWN, buff=0.5)

        apply_bayes_eq = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)p_\theta(x_0|x_1)\prod_{t=2}^T p_\theta(x_{t-1}|x_t)}{q(x_1|x_0)\prod_{t=2}^T },x_0)}\right]",
            font_size=50
        )
        
        self.play(Write(desc))
        self.wait(1)

        self.play(Write(eq1))
        self.wait(1)
        self.play(Write(eq2))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(eq1), FadeOut(eq2))

    def show_bayes_substitution(self, x0_conditioning_eq):
        # Show equation 57 - separated form
        eq_57 = MathTex(
            r"elbo = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)p_\theta(x_0|x_1)}{q(x_1|x_0)}\right]",
            r" + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1},x_0)}\right]",
            font_size=40
        ).shift(UP)
        
        eq_57[0].set_color(BLUE)  # E1 term
        eq_57[1].set_color(ORANGE)  # E2 term
        
        # self.play(Write(eq_57))
        self.play(ReplacementTransform(x0_conditioning_eq, eq_57))
        self.wait(2)
        
        # Show equation 58 - E1 + E2
        eq_58 = MathTex(
            r"= \hspace{1cm}      E_1      \hspace{2cm}    +       \hspace{2cm}           E_2",
            font_size=40
        ).next_to(eq_57, DOWN, buff=0.5)
        
        # Add labels
        e1_label = Text("E1: Edge terms (x0, x1, xT)", font_size=24, color=BLUE).next_to(eq_58, DOWN, buff=0.5).shift(LEFT*3)
        e2_label = Text("E2: Middle terms (t=2 to T)", font_size=24, color=ORANGE).next_to(eq_58, DOWN, buff=0.5).shift(RIGHT*3)
        
        self.play(Write(eq_58))
        self.play(FadeIn(e1_label), FadeIn(e2_label))
        self.wait(2)
        
        self.play(FadeOut(e1_label), FadeOut(e2_label), FadeOut(eq_58))

        return eq_57

    def show_focus_on_e2(self, eq_57):
        # desc = Text("Focusing on E2: The Middle Terms", font_size=36, color=YELLOW)
        # desc.to_edge(UP)
        
        #explanation = Text(
        #    "We'll focus on E2 because that's where the complexity is.\n"
        #    "E1 contains the boundary terms—we'll return to it later.\n"
        #    "This separation keeps our equations manageable.",
        #    font_size=28,
        #    color=WHITE
        #).shift(UP*0.5)
        
        eq_e2 = MathTex(
            r"E_2 = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1},x_0)}\right]",
            font_size=50,
            color=ORANGE
        )
        
        # self.play(Write(desc))
        # self.play(FadeIn(explanation, shift=UP))
        # self.wait(2)

        self.play(ReplacementTransform(eq_57, eq_e2))
        self.wait(2)
        # self.play(Write(eq_e2))
        # self.wait(2)
        # self.play(FadeOut(desc), FadeOut(explanation), FadeOut(eq_e2))

        return eq_e2

    def show_bayes_on_e2(self, eq_e2):
        desc = Text("Applying Bayes Rule to E2", font_size=30, color=GREEN)
        desc.next_to(eq_e2, DOWN, buff=0.5)
        
        #eq1 = MathTex(
        #    r"E_2 = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1},x_0)}\right]",
        #    font_size=50
        #)
        
        eq2 = MathTex(
            r"E2 = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)q(x_t|x_0)/q(x_{t-1}|x_0)}\right]",
            font_size=50
        )

        eq60 = MathTex(
            r"E2 = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)q(x_{t-1}|x_0)}{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}\right]",
            font_size=50
        )
        
        self.play(Write(desc))
        self.wait(1)

        # self.play(Write(eq1))
        self.play(ReplacementTransform(eq_e2, eq2))
        self.wait(2)

        self.play(FadeOut(desc))
        self.wait(1)

        self.play(ReplacementTransform(eq2, eq60))
        self.wait(2)

        # self.play(Write(eq2))
        # self.wait(2)

        # self.play(FadeOut(desc), FadeOut(eq1), FadeOut(eq2))
        return eq60

    def show_term_separation(self, eq_e2):
        desc = Text("Separating Logarithm Terms", font_size=30, color=GREEN)
        # desc.to_edge(UP)
        desc.next_to(eq_e2, DOWN, buff=0.5)
        
        eq = MathTex(
            r"E2 = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} + \log \prod_{t=2}^T \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}\right]",
            font_size=50
        )
        
        self.play(Write(desc))
        self.wait(1)

        self.play(ReplacementTransform(eq_e2, eq))
        # self.play(Write(eq))
        self.wait(2)

        self.play(FadeOut(desc))
        self.wait(1)

        return eq
 
    def show_cancellation_animation(self, e2):
        # Create the individual fraction terms
        frac1 = MathTex(r"\frac{q(x_1|x_0)}{q(x_2|x_0)}", font_size=48, color=WHITE)
        times1 = MathTex(r"\cdot", font_size=48)
        frac2 = MathTex(r"\frac{q(x_2|x_0)}{q(x_3|x_0)}", font_size=48, color=WHITE)
        times2 = MathTex(r"\cdot", font_size=48)
        frac3 = MathTex(r"\frac{q(x_3|x_0)}{q(x_4|x_0)}", font_size=48, color=WHITE)
        times3 = MathTex(r"\cdots", font_size=48)
        frac4 = MathTex(r"\frac{q(x_{T-1}|x_0)}{q(x_T|x_0)}", font_size=48, color=WHITE)
        
        # Arrange terms horizontally
        fractions = VGroup(frac1, times1, frac2, times2, frac3, times3, frac4)
        fractions.arrange(RIGHT, buff=0.2)
        # fractions.move_to(ORIGIN)
        fractions.next_to(e2, DOWN, buff=0.5)
        
        self.play(Write(fractions))
        self.wait(1)
        
        # Highlight q(x_2|x_0) in both places
        print('size of the rectangle')
        print(len(frac1[0]))
        box1 = SurroundingRectangle(frac1[0][9:17], color=RED, buff=0.05)  # denominator of first fraction
        box2 = SurroundingRectangle(frac2[0][0:8], color=RED, buff=0.05)   # numerator of second fraction
        
        cancel_text1 = Text("q(x₂|x₀) cancels!", font_size=24, color=RED)
        cancel_text1.next_to(fractions, DOWN, buff=0.5)
        
        self.play(Create(box1), Create(box2))
        self.play(FadeIn(cancel_text1, shift=UP))
        self.wait(1)
        
        # Cross out q(x_2|x_0)
        cross1 = Line(box1.get_corner(DL), box1.get_corner(UR), color=RED, stroke_width=4)
        cross2 = Line(box2.get_corner(DL), box2.get_corner(UR), color=RED, stroke_width=4)
        
        self.play(Create(cross1), Create(cross2))
        self.wait(1)
        self.play(FadeOut(box1), FadeOut(box2), FadeOut(cancel_text1), FadeOut(cross1), FadeOut(cross2))
        
        # Highlight q(x_3|x_0)
        box3 = SurroundingRectangle(frac2[0][9:17], color=BLUE, buff=0.05)  # denominator of second fraction
        box4 = SurroundingRectangle(frac3[0][0:8], color=BLUE, buff=0.05)   # numerator of third fraction
        
        cancel_text2 = Text("q(x₃|x₀) cancels!", font_size=24, color=BLUE)
        cancel_text2.next_to(fractions, DOWN, buff=0.5)
        
        self.play(Create(box3), Create(box4))
        self.play(FadeIn(cancel_text2, shift=UP))
        self.wait(1)
        
        # Cross out q(x_3|x_0)
        cross3 = Line(box3.get_corner(DL), box3.get_corner(UR), color=BLUE, stroke_width=4)
        cross4 = Line(box4.get_corner(DL), box4.get_corner(UR), color=BLUE, stroke_width=4)
        
        self.play(Create(cross3), Create(cross4))
        self.wait(1)
        self.play(FadeOut(box3), FadeOut(box4), FadeOut(cancel_text2), FadeOut(cross3), FadeOut(cross4))
        
        # Show pattern continues
        dots_text = Text("Pattern continues...", font_size=28, color=YELLOW)
        dots_text.next_to(fractions, DOWN, buff=0.5)
        
        self.play(FadeIn(dots_text, shift=UP))
        self.wait(1)
        self.play(FadeOut(dots_text))
        
        # Show final result - only boundary terms remain
        result_text = Text("Only boundary terms survive!", font_size=32, color=GREEN)
        result_text.next_to(fractions, DOWN, buff=1)
        
        self.play(FadeIn(result_text, shift=UP))
        self.wait(1)
        
        # Show final simplified form
        final_eq = MathTex(
            r"= \frac{q(x_1|x_0)}{q(x_T|x_0)}",
            font_size=56,
            color=GOLD
        ).shift(DOWN*2)
        
        self.play(ReplacementTransform(fractions, final_eq))
        self.wait(2)
        
        # self.play(FadeOut(desc), FadeOut(result_text), FadeOut(final_eq))
        self.play(FadeOut(result_text), FadeOut(final_eq))

    def show_cancellation(self, e2_term_separation):
        desc = Text("After cancellation", font_size=30, color=GREEN)
        # desc.to_edge(UP)
        desc.next_to(e2_term_separation, DOWN, buff=1.0)
        
        eq = MathTex(
            r"E2 = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} + \log \prod_{t=2}^T \frac{q(x_1|x_0)}{q(x_T|x_0)}\right]",
            font_size=50
        )
        
        self.play(Write(desc))
        self.wait(1)

        self.play(ReplacementTransform(e2_term_separation, eq))
        self.wait(2)

        self.play(FadeOut(desc), FadeOut(eq))
        self.wait(1)

    def show_three_components(self):
        # Show equation 57 - separated form
        eq_57 = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)p_\theta(x_0|x_1)}{q(x_1|x_0)}\right]",
            r" + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1},x_0)}\right]",
            font_size=40
        )

        desc = Text("Replacing E2 in ELBO equation", font_size=36, color=GREEN)
        desc.next_to(eq_57, DOWN, buff=0.5)
        # desc.to_edge(UP)
        
        eq = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T})|x_0}[",
            r"\log \frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}",
            r" + \log \frac{q(x_1|x_0)}{q(x_T|x_0)}",
            r" + \log \prod_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}",
            r"]",
            font_size=36
        )
        
        # Color different components
        eq[1].set_color(BLUE)
        eq[2].set_color(GREEN)
        eq[3].set_color(PURPLE)

        eq_67 = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[ \log \frac{p(x_T) p_\theta(x_0|x_1)}{q(x_T|x_0)} + \sum_{t=2}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right]",
            font_size=40
        )

        self.play(Write(eq_57))
        self.wait(2)

        self.play(Write(desc))
        self.wait(1)
        
        self.play(ReplacementTransform(eq_57, eq))
        self.wait(3)

        self.play(FadeOut(desc))
        self.wait(1)

        box3 = SurroundingRectangle(eq[1][18:], color=BLUE, buff=0.05)  # denominator of second fraction
        box4 = SurroundingRectangle(eq[2][4:12], color=BLUE, buff=0.05)   # numerator of third fraction
        
        cancel_text2 = Text("q(x1|x₀) cancels!", font_size=24, color=BLUE)
        cancel_text2.next_to(eq, DOWN, buff=0.5)
        
        self.play(FadeIn(cancel_text2, shift=UP))
        self.play(Create(box3), Create(box4))
        self.wait(1)

        self.play(FadeOut(box3), FadeOut(box4), FadeOut(cancel_text2))
        self.wait(1)

        self.play(ReplacementTransform(eq, eq_67))
        self.wait(2)

        self.play(FadeOut(eq_67))
        self.wait(1)

    def show_linearity(self):
        # desc.to_edge(UP)
        
        eq1 = MathTex(
            r"\mathbb{E}[A + B + C] = \mathbb{E}[A] + \mathbb{E}[B] + \mathbb{E}[C]",
            font_size=70
        )

        desc = Text("Applying Linearity of Expectations", font_size=36, color=GREEN)
        desc.next_to(eq1, DOWN, buff=0.5)

        eq_68 = MathTex(
            r"ELBO = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log p_\theta(x_0|x_1)\right] + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_0)}\right] + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\sum_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right]",
            font_size=30
        )

        eq_69 = MathTex(
            r"ELBO = \mathbb{E}_{q(x_1|x_0)}\left[\log p_\theta(x_0|x_1)\right] + \mathbb{E}_{q(x_T|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_0)}\right] + \mathbb{E}_{q(x_{1:T}|x_0)}\left[\sum_{t=2}^T \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right]",
            font_size=30
        )
        
        eq2 = MathTex(
            r"ELBO = L_0 + L_T + \sum_{t=2}^T L_{t-1}",
            font_size=80
        )
        
        self.play(Write(eq1))
        self.play(Write(desc))
        self.wait(1)

        self.play(ReplacementTransform(eq1, eq_68))
        # self.play(Write(eq2))
        self.wait(2)

        self.play(FadeOut(desc))
        self.wait(1)

        self.play(ReplacementTransform(eq_68, eq2))
        self.wait(2)

        return eq2

    def show_final_decomposition(self, linear_eq):
        desc = Text("Final ELBO Decomposition", font_size=36, color=GREEN)
        # desc.to_edge(UP)
        desc.next_to(linear_eq, DOWN, buff=0.5)
        
        eq = MathTex(
            r"ELBO = L_0 + L_T + \sum_{t=2}^T L_{t-1}",
            font_size=100,
            color=GOLD
        )
        
        self.play(Write(desc))
        self.play(ReplacementTransform(linear_eq, eq))
        # self.play(Write(eq))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(eq))

    def explain_l0(self):
        desc = Text("L₀: Reconstruction Term", font_size=36, color=BLUE)
        desc.to_edge(UP)
        
        eq = MathTex(
            r"L_0 = \mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]",
            font_size=100,
            color=BLUE
        ).shift(UP*0.5)
        
        exp = Text(
            "How well we reconstruct the final image x₀\n"
            "from the first denoising step\n"
            "Optimized using Monte Carlo estimate",
            font_size=28,
            color=WHITE
        ).shift(DOWN*1.5)
        
        self.play(Write(desc))
        self.play(Write(eq))
        self.wait(1)
        self.play(FadeIn(exp, shift=UP))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(eq), FadeOut(exp))

    def explain_lt(self):
        desc = Text("LT: Prior Matching Term", font_size=36, color=GREEN)
        desc.to_edge(UP)
        
        eq1 = MathTex(
            r"L_T = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p(x_T)}{q(x_T|x_0)}\right]",
            font_size=70,
            color=GREEN
        ).shift(UP*1)
        
        eq2 = MathTex(
            r"= D_{KL}(q(x_T|x_0) \parallel p(x_T))",
            font_size=70,
            color=GREEN
        ).shift(DOWN*0.5)
        
        exp = Text(
            "KL divergence between final noisy distribution\n"
            "and Gaussian prior\n"
            "No trainable parameters → derivative = 0",
            font_size=26,
            color=WHITE
        ).shift(DOWN*2.5)
        
        self.play(Write(desc))
        self.play(Write(eq1))
        self.wait(1)
        self.play(Write(eq2))
        self.wait(1)
        self.play(FadeIn(exp, shift=UP))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(eq1), FadeOut(eq2), FadeOut(exp))

    def explain_sum_terms(self):
        desc = Text("∑Lt-1: Denoising Matching Terms", font_size=36, color=PURPLE)
        desc.to_edge(UP)
        
        eq1 = MathTex(
            r"\sum_{t=2}^T L_{t-1} = \sum_{t=2}^T \mathbb{E}_{q(x_t,x_{t-1}|x_0)}\left[\log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right]",
            font_size=50,
            color=PURPLE
        ).shift(UP*1)
        
        eq2 = MathTex(
            r"= \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) \parallel p_\theta(x_{t-1}|x_t))",
            font_size=50,
            color=PURPLE
        ).shift(DOWN*0.8)
        
        exp = Text(
            "Sum of KL divergences at each timestep\n"
            "Compares model's denoiser to perfect denoising\n"
            "This is where the model learns!",
            font_size=26,
            color=WHITE
        ).shift(DOWN*2.5)
        
        self.play(Write(desc))
        self.play(Write(eq1))
        self.wait(1)
        self.play(Write(eq2))
        self.wait(1)
        self.play(FadeIn(exp, shift=UP))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(eq1), FadeOut(eq2), FadeOut(exp))

    def show_why_matters(self):
        desc = Text("Why This Decomposition Matters", font_size=38, color=YELLOW)
        desc.to_edge(UP)
        
        reasons = VGroup(
            Text("✓ Each expectation depends on fewer variables", font_size=28, color=GREEN),
            Text("✓ Lower variance estimates", font_size=28, color=GREEN),
            Text("✓ Each term has clear interpretation", font_size=28, color=GREEN),
            Text("✓ Easier to optimize computationally", font_size=28, color=GREEN),
            Text("✓ Can focus on important timesteps", font_size=28, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        self.play(Write(desc))
        self.play(FadeIn(reasons, shift=UP, lag_ratio=0.3))
        self.wait(3)
        self.play(FadeOut(desc), FadeOut(reasons))

    def show_final_summary(self):
        title = Text("Complete ELBO Decomposition", font_size=40, color=GOLD)
        title.to_edge(UP)
        
        eq = MathTex(
            r"ELBO = ",
            r"\underbrace{\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]}_{L_0: \text{ Reconstruction}}",
            r" + ",
            r"\underbrace{D_{KL}(q(x_T|x_0) \parallel p(x_T))}_{L_T: \text{ Prior Matching}}",
            r" + ",
            r"\underbrace{\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) \parallel p_\theta(x_{t-1}|x_t))}_{\sum L_{t-1}: \text{ Denoising Steps}}",
            font_size=30
        )
        eq[1].set_color(BLUE)
        eq[3].set_color(GREEN)
        eq[5].set_color(PURPLE)
        
        self.play(Write(title))
        self.play(Write(eq))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(eq))


# To render this animation, save as ddpm_elbo.py and run:
# python -m manim -pqk ddpm.py DDPMELBODecomposition
# 
# Quality options:
# -ql : low quality (854x480 15fps)
# -qm : medium quality (1280x720 30fps)
# -qh : high quality (1920x1080 60fps)
# -qk : 4K quality (3840x2160 60fps)
#
# Add -p to preview after rendering
# Add --format=gif to create an animated GIF

from manim import *

class DDPMGaussianSampling(Scene):
    def construct(self):
        self.camera.background_color = "#0f0f23"
        
        # Title
        title = self.show_title()
        
        # Part 1: Gaussian Parameters Derivation
        eq_74, eq_21 = self.show_bayes_flip()
        eq_75 = self.show_posterior_as_gaussians(eq_74, eq_21)
        self.show_gaussian_pdf(eq_75)
        prop_eq = self.show_expanding_terms(eq_75)
        eq_79 = self.show_completing_square(prop_eq)
        eq_80_81 = self.show_grouping_terms(eq_79)
        eq_82 = self.show_removing_constant(eq_80_81)

        eq_85 = self.show_factoring_denominators(eq_82)

        self.show_simplification(eq_85)

        self.convert_to_gaussian()

        self.show_final_gaussian_form()

        self.show_reparameterization()

        eq_97 = self.show_mean_derivation()

        eq_99 = self.show_expanding_and_separating(eq_97)

        eq_101 = self.show_cancellation_steps(eq_99)

        self.show_final_mean_form(eq_101)
        
        # ## Part 2: Sampling Algorithm
        self.show_sampling_algorithm()
        self.show_algorithm_walkthrough()
        
    def show_title(self):
        title = Text("DDPM: Gaussian Parameters & Sampling", font_size=48, color=BLUE)
        subtitle = Text("From KL Divergence to Practical Algorithm", font_size=32, color=PURPLE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        return title
    
    def show_bayes_flip(self):
        # desc.to_edge(UP)

        eq_73 = MathTex(
            r"\sum_{t=2}^T L_{t-1} = \sum_{t=2}^T \mathbb{E}_{q(x_t,x_{t-1}|x_0)}\left[\log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\right]",
            font_size=50,
            color=PURPLE
        ).shift(UP*1)

        desc1 = Text("Perfect denoising process in the equation", font_size=32, color=GREEN)
        desc1.next_to(eq_73, DOWN, buff=0.5)

        print(len(eq_73[0]))
        box1 = SurroundingRectangle(eq_73[0][45:58], color=RED, buff=0.05)  # denominator of first fraction
        
        eq_74 = MathTex(
            r"q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}",
            font_size=60
        )
        eq_74.next_to(eq_73, DOWN, buff=1)

        desc2 = Text("Flipping the Posterior using Bayes Rule", font_size=30, color=GREEN)
        desc2.next_to(eq_74, DOWN, buff=0.5)

        self.play(Write(eq_73))
        self.wait(1)
        
        self.play(Write(desc1))
        self.play(Write(box1))
        self.wait(1)

        self.play(FadeOut(desc1))
        self.play(Write(eq_74))
        self.wait(1)

        self.play(Write(desc2))
        self.wait(2)

        self.play(FadeOut(desc2), FadeOut(eq_73), FadeOut(box1))

        self.play(eq_74.animate.shift(UP * 3))

        eq_21 = MathTex(
            r"q(x_t}|x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_0, (1-\alpha_t)I)",
            font_size=40
        )
        eq_21.next_to(eq_74, DOWN, buff=1)

        self.play(Write(eq_21))
        self.wait(2)

        return eq_74, eq_21
    
    def show_gaussian_pdf(self, eq_74):
        desc = Text("Gaussian Probability Density Function", font_size=30, color=GREEN)
        desc.next_to(eq_74, DOWN, buff=0.5)
        # desc.to_edge(UP)
        
        eq_76 = MathTex(
            r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
            font_size=30
        )
        eq_76.next_to(desc, DOWN, buff=0.5)
        
        self.play(Write(desc))
        self.play(Write(eq_76))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(eq_76))

        # return eq_76
    
    def show_posterior_as_gaussians(self, eq_74, eq_76):
        # desc.to_edge(UP)
        
        eq_75 = MathTex(
            r"q(x_{t-1}|x_t, x_0) = \frac{\mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I) \mathcal{N}(x_{t-1}; \sqrt{\overline{\alpha_{t-1}}}x_0, (1-\overline{\alpha_{t-1}})I)}{\mathcal{N}(x_t; \sqrt{\overline{\alpha_t}}x_0, (1-\overline{\alpha_t})I)}",
            font_size=40
        )

        desc = Text("Writing Posterior as Product of Gaussians", font_size=30, color=GREEN)
        desc.next_to(eq_75, DOWN, buff=1.0)
        
        self.play(ReplacementTransform(eq_74, eq_75))
        # self.play(Write(eq_75))
        self.play(FadeOut(eq_76))
        self.wait(1)
        self.play(Write(desc))
        self.wait(3)
        self.play(FadeOut(desc))

        return eq_75
    
    def show_expanding_terms(self, eq_75):
        desc = Text("Expanding as Exponentials", font_size=30, color=GREEN)
        desc.next_to(eq_75, DOWN, buff=2)
        
        eq_77_1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) = ",
            font_size=40
        ).shift(UP*3)
        eq_77_2 = MathTex(
            r"\frac{1}{\sqrt{2\pi(1-\alpha_t)}} \exp\left(-\frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{2(1-\alpha_t)}\right)",
            font_size=40
        )#.shift(UP*0.5)
        eq_77_2.next_to(eq_77_1, DOWN, buff=0.5)
        eq_77_3 = MathTex(
            r" \times \frac{1}{\sqrt{2\pi(1-\overline{\alpha_{t-1}})}} \exp\left(-\frac{(x_{t-1} - \sqrt{\overline{\alpha_{t-1}}}x_0)^2}{2(1-\overline{\alpha_{t-1}})}\right)",
            font_size=40
        )#.shift(UP*0.5)
        eq_77_3.next_to(eq_77_2, DOWN, buff=0.5)
        eq_77_4 = MathTex(
            r" / \frac{1}{\sqrt{2\pi(1-\overline{\alpha_t})}} \exp\left(-\frac{(x_t - \sqrt{\overline{\alpha_t}}x_0)^2}{2(1-\overline{\alpha_t})}\right)",
            font_size=40
        )#.shift(UP*0.5)
        eq_77_4.next_to(eq_77_3, DOWN, buff=0.5)
        
        # self.play(Write(desc))
        self.play(ReplacementTransform(eq_75, eq_77_1))
        self.play(Write(desc))
        self.play(Write(eq_77_2))
        self.play(Write(eq_77_3))
        self.play(Write(eq_77_4))
        self.play(FadeOut(desc))
        self.wait(3)
        
        # Highlight that it's proportional
        prop_eq = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp\left(-\left[\frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{2(1-\alpha_t)} + \frac{(x_{t-1} - \sqrt{\overline{\alpha_{t-1}}}x_0)^2}{2(1-\overline{\alpha_{t-1}})} - \frac{(x_t - \sqrt{\overline{\alpha_t}}x_0)^2}{2(1-\overline{\alpha_t})}\right]\right)",
            font_size=32
        )
        prop_eq.next_to(eq_77_4, DOWN, buff=0.5)
        
        self.play(Write(prop_eq))
        self.wait(2)

        self.play(FadeOut(eq_77_1), FadeOut(eq_77_2), FadeOut(eq_77_3), FadeOut(eq_77_4))
        self.wait(1)

        self.play(prop_eq.animate.shift(UP * 3))
        self.wait(1)

        return prop_eq
    
    def show_completing_square(self, prop_eq):
        # desc = Text("Expanding Squared Terms: (a-b)² = a² - 2ab + b²", font_size=36, color=YELLOW)
        # desc.to_edge(UP)
        
        # Show the algebraic identity visually
        identity = MathTex(
            r"(a - b)^2 = a^2 - 2ab + b^2",
            font_size=32,
            color=GREEN
        ).shift(DOWN*3)
        
        # self.play(Write(desc))
        self.play(Write(identity))
        self.wait(2)
        
        # Show equation 79
        eq_79 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp\left(-\frac{1}{2}\left[\frac{x_t^2 + \alpha_tx_{t-1}^2 - 2\sqrt{\alpha_t}x_tx_{t-1}}{(1-\alpha_t)} + \frac{x_{t-1}^2 + \overline{\alpha_{t-1}}x_0^2 - 2\sqrt{\overline{\alpha_{t-1}}}x_{t-1}x_0}{(1-\overline{\alpha_{t-1}})} - \frac{x_t^2 + \overline{\alpha_t}x_0^2 - 2\sqrt{\overline{\alpha_t}}x_tx_0}{(1-\overline{\alpha_t})}\right]\right)",
            font_size=25
        )

        eq_79_1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp\left(-\frac{1}{2}\left[\frac{x_t^2 + \alpha_tx_{t-1}^2 - 2\sqrt{\alpha_t}x_tx_{t-1}}{(1-\alpha_t)}",
            font_size=50
        )
        eq_79_1.to_edge(UP)

        eq_79_2 = MathTex(
            r"+ \frac{x_{t-1}^2 + \overline{\alpha_{t-1}}x_0^2 - 2\sqrt{\overline{\alpha_{t-1}}}x_{t-1}x_0}{(1-\overline{\alpha_{t-1}})}",
            font_size=60
        )
        eq_79_2.next_to(eq_79_1, DOWN, buff=0.5)

        eq_79_3 = MathTex(
            r"- \frac{x_t^2 + \overline{\alpha_t}x_0^2 - 2\sqrt{\overline{\alpha_t}}x_tx_0}{(1-\overline{\alpha_t})}\right]\right)",
            font_size=60
        )
        eq_79_3.next_to(eq_79_2, DOWN, buff=0.5)
        eq_79 = VGroup(eq_79_1, eq_79_2, eq_79_3)

        
        # self.play(Write(eq_79))
        self.play(ReplacementTransform(prop_eq, eq_79))
        self.play(FadeOut(identity))
        self.wait(3)
        # self.play(FadeOut(identity), FadeOut(eq_79))

        return eq_79

    def show_grouping_terms(self, eq_79):
        # Breaking equation into separate lines
        line1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp\left(-\frac{1}{2}\left[\right.\right.",
            font_size=50
        )
        line1.to_edge(UP)
    
        line2 = MathTex(
            r"\frac{\alpha_tx_{t-1}^2 - 2\sqrt{\alpha_t}x_tx_{t-1}}{(1-\alpha_t)}",
            font_size=50
        )
    
        line3 = MathTex(
            r"+ \frac{x_{t-1}^2 - 2\sqrt{\overline{\alpha_{t-1}}}x_{t-1}x_0}{(1-\overline{\alpha_{t-1}})}",
            font_size=50
        )
    
        line4 = MathTex(
            r"+ C(x_t, x_0) \left.\left.\right]\right)",
            font_size=50
        )
    
        # Group them vertically
        eq_80_81 = VGroup(line1, line2, line3, line4).arrange(
            DOWN, center=False, aligned_edge=LEFT, buff=0.3
        )
    
        desc = Text("Grouping Terms with x_t and x_0", font_size=36, color=YELLOW)
        desc.next_to(eq_79, DOWN, buff=0.5)
    
        c_note = Text(
            "C(x_t, x_0) contains all terms with x_t and x_0 only",
            font_size=24,
            color=ORANGE
        ).shift(DOWN*2)
    
        self.play(Write(desc))
        self.wait(1)
        self.play(ReplacementTransform(eq_79, eq_80_81))
        self.wait(2)
        self.play(FadeOut(desc))
        self.play(FadeIn(c_note, shift=UP))
        self.wait(2)
        self.play(FadeOut(c_note))
        return eq_80_81
    
    def show_removing_constant(self, eq_80_81):
        desc = Text("Removing Constant: e^(x+C) = Ce^x ∝ e^x", font_size=30, color=YELLOW)
        desc.next_to(eq_80_81, DOWN, buff=0.5)
    
        prop_property = MathTex(
            r"e^{(x+C)} = e^x \cdot e^C = Ce^x \propto e^x",
            font_size=30,
            color=RED
        ).next_to(desc, DOWN, buff=0.5)
    
        # Breaking equation into separate lines
        line1 = MathTex(
            r"\Rightarrow q(x_{t-1}|x_t, x_0) \propto \exp\left(-\frac{1}{2}\left[\right.\right.",
            font_size=50
        )
        line1.to_edge(UP)
    
        line2 = MathTex(
            r"\frac{\alpha_tx_{t-1}^2 - 2\sqrt{\alpha_t}x_tx_{t-1}}{(1-\alpha_t)} + \frac{x_{t-1}^2 - 2\sqrt{\overline{\alpha_{t-1}}}x_{t-1}x_0}{(1-\overline{\alpha_{t-1}})}",
            font_size=50
        )

        # If line2 is still too long, break it into two fractions:
        line2a = MathTex(
            r"\frac{\alpha_tx_{t-1}^2 - 2\sqrt{\alpha_t}x_tx_{t-1}}{(1-\alpha_t)}",
            font_size=50
        )

        line2b = MathTex(
            r"+ \frac{x_{t-1}^2 - 2\sqrt{\overline{\alpha_{t-1}}}x_{t-1}x_0}{(1-\overline{\alpha_{t-1}})} \left.\left.\right]\right)",
            font_size=50
        )

        # Group them vertically
        eq_82 = VGroup(line1, line2a, line2b).arrange(DOWN, center=False, aligned_edge=LEFT, buff=0.3)
    
        self.play(Write(desc))
        self.play(Write(prop_property))
        self.wait(2)
        self.play(ReplacementTransform(eq_80_81, eq_82))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(prop_property))
        return eq_82

    def show_factoring_denominators(self, eq_82):
        desc = Text("Finding Common Denominators", font_size=36, color=GREEN)
        # desc.to_edge(UP)
        desc.next_to(eq_82, DOWN, buff=2.0)

        # eq_83 = MathTex(
        #     r"q(x_{t-1}|x_t, x_0) \propto \exp \left( -\frac{1}{2} \left[ \frac{\alpha_t x_{t-1}^2}{(1 - \alpha_t)} + \frac{x_{t-1}^2}{(1 - \overline{\alpha_{t-1}})} + \frac{-2\sqrt{\alpha_t} x_t x_{t-1}}{(1 - \alpha_t)} + \frac{-2\sqrt{\overline{\alpha_{t-1}}} x_{t-1} x_0}{(1 - \overline{\alpha_{t-1}})} \right] \right)",
        #     font_size=32
        # )
        eq_83_1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp \left( -\frac{1}{2} ",
            font_size=50
        ).to_edge(UP)
        eq_83_2 = MathTex(
            r"\left[ \frac{\alpha_t x_{t-1}^2}{(1 - \alpha_t)} + \frac{x_{t-1}^2}{(1 - \overline{\alpha_{t-1}})} ",
            font_size=50
        )
        eq_83_3 = MathTex(
            r"+ \frac{-2\sqrt{\alpha_t} x_t x_{t-1}}{(1 - \alpha_t)} + \frac{-2\sqrt{\overline{\alpha_{t-1}}} x_{t-1} x_0}{(1 - \overline{\alpha_{t-1}})} \right] \right)",
            font_size=50
        )
        eq_83 = VGroup(eq_83_1, eq_83_2, eq_83_3).arrange(DOWN, center=False, aligned_edge=LEFT, buff=0.3)

        eq_84 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp \left( -\frac{1}{2} \left[ \left( \frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \overline{\alpha_{t-1}}} \right) x_{t-1}^2 - 2 \left( \frac{\sqrt{\alpha_t} x_t }{1 - \alpha_t} + \frac{\sqrt{\overline{\alpha_{t-1}}} x_0}{1 - \overline{\alpha_{t-1}}} \right) x_{t-1} \right] \right)",
            font_size=32
        )

        eq_85_1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp \left( -\frac{1}{2} \left[ ",
            font_size=50
        ).to_edge(UP)
        eq_85_2 = MathTex(
            r" \frac{\alpha_t (1 - \overline{\alpha_{t-1}}) + 1 - \alpha_t}{(1 - \alpha_t)(1 - \overline{\alpha_{t-1}})} x_{t-1}^2 ",
            font_size=50
        )
        eq_85_3 = MathTex(
            r"- 2 \left( \frac{\sqrt{\alpha_t} x_t (1 - \overline{\alpha_{t-1}}) + (1 - \alpha_t)\sqrt{\overline{\alpha_{t-1}}} x_0 }{(1 - \alpha_t) (1 - \overline{\alpha_{t-1}})} \right) x_{t-1} \right] \right)",
            font_size=50
        )
        # eq_85 = VGroup(eq_85_1, eq_85_2, eq_85_3).arrange(DOWN, center=False, aligned_edge=LEFT, buff=0.3)
        eq_85 = VGroup(eq_85_1, eq_85_2, eq_85_3).arrange(DOWN, buff=0.3)
        
        # eq_83_85 = MathTex(
        #     r"= \exp\left(-\frac{1}{2}\left[\frac{\alpha_t(1-\overline{\alpha_{t-1}}) + 1-\alpha_t}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}x_{t-1}^2 - 2\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t)x_0}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}x_{t-1}\right]\right)",
        #     font_size=22
        # )
        
        self.play(Write(desc))
        self.wait(1)

        self.play(ReplacementTransform(eq_82, eq_83))
        self.wait(2)

        self.play(ReplacementTransform(eq_83, eq_84))
        self.wait(2)
        self.play(FadeOut(eq_84))

        # self.play(ReplacementTransform(eq_84, eq_83_85))
        self.play(Write(eq_85))
        self.wait(2)
        # self.play(FadeOut(eq_85))

        self.play(FadeOut(desc))
        self.wait(2)

        # return eq_83_85
        return eq_85
    
    def show_simplification(self, eq_85):
        desc = Text("Simplifying", font_size=32, color=GREEN)
        # desc.to_edge(UP)
        desc.next_to(eq_85, DOWN, buff=2.)
        
        simplification = MathTex(
            r"\alpha_t - \alpha_t\overline{\alpha_{t-1}} + 1 - \alpha_t = 1 - \overline{\alpha_t}",
            font_size=32,
            color=GREEN
        )
        simplification.next_to(desc, DOWN, buff=0.5)
        
        eq_87_1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp\left(-\frac{1}{2}",
            font_size=50
        )
        eq_87_1.to_edge(UP)
        eq_87_2 = MathTex(
            r"\left[\frac{1-\overline{\alpha_t}}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}x_{t-1}^2 ",
            font_size=50
        )
        eq_87_3 = MathTex(
            r"- 2\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t)x_0}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}x_{t-1}\right]\right)",
            font_size=50
        )
        eq_87 = VGroup(eq_87_1, eq_87_2, eq_87_3).arrange(DOWN, buff=0.3)
        eq_87.to_edge(UP)
        
        self.play(Write(desc))
        self.play(Write(simplification))
        self.wait(2)

        # self.play(Write(eq_87))
        self.play(ReplacementTransform(eq_85, eq_87))
        self.wait(2)

        self.play(FadeOut(desc), FadeOut(simplification), FadeOut(eq_87))

    def convert_to_gaussian(self):
        desc = Text('Writing in the exp form', font_size=36, color=GREEN)
        desc.to_edge(DOWN)

        eq_88_1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp \left( -\frac{1}{2} \left( \frac{1 - \overline{\alpha_t}}{(1 - \alpha_t)(1 - \overline{\alpha_{t-1}})} \right) ",
            font_size=60,
        )
        eq_88_2 = MathTex(
            r"\left[ x_{t-1}^2 - 2 \frac{\sqrt{\alpha_t}(1 - \overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1 - \alpha_t)x_0}{1 - \overline{\alpha_t}}x_{t-1} \right] \right)",
            font_size=60,
        )
        eq_88 = VGroup(eq_88_1, eq_88_2).arrange(DOWN, buff=0.5)

        eq_89_1 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \exp\left(-\frac{1}{2 \frac{(1 - \alpha_t)(1 - \overline{\alpha_{t-1}})}{1  - \overline{\alpha_t}}} \left[x_{t-1}^2 ",
            font_size=60
        )
        eq_89_2 = MathTex(
            r"- \frac{\sqrt{\alpha_t}(1 - \overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1 - \alpha_t)x_0}{1 - \overline{\alpha_t}} x_{t-1} \right]\right)",
            font_size=60
        )
        eq_89 = VGroup(eq_89_1, eq_89_2).arrange(DOWN, buff=0.5)

        self.play(Write(desc))
        self.wait(1)

        self.play(Write(eq_88))
        self.wait(2)
        self.play(FadeOut(eq_88))

        self.play(Write(eq_89))
        self.wait(2)

        self.play(FadeOut(desc), FadeOut(eq_89))
    
    def show_final_gaussian_form(self):
        desc = Text("Recognizing Gaussian Form: exp(-½σ⁻²(x-μ)²)", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        eq_90 = MathTex(
            r"q(x_{t-1}|x_t, x_0) \propto \mathcal{N}\left(x_{t-1}; \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t)x_0}{1-\overline{\alpha_t}}, \frac{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_t}}I\right)",
            font_size=26
        ).shift(UP*0.5)
        
        eq_91 = MathTex(
            r"\mu_q(x_t, x_0) := \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t)x_0}{1-\overline{\alpha_t}}",
            font_size=34,
            color=BLUE
        ).shift(DOWN*1)
        
        eq_92 = MathTex(
            r"\Sigma_q(t) := \frac{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_t}}I\right)",
            font_size=38,
            color=GREEN
        ).shift(DOWN*2.2)
        
        self.play(Write(desc))
        self.play(Write(eq_90))
        self.wait(2)
        self.play(Write(eq_91))
        self.play(Write(eq_92))
        self.wait(3)
        self.play(FadeOut(desc), FadeOut(eq_90), FadeOut(eq_91), FadeOut(eq_92))
    
    def show_reparameterization(self):
        desc = Text("Reparameterizing: Solving for x_0 from x_t", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        eq_14 = MathTex(
            r"x_t = \sqrt{\overline{\alpha_t}} \cdot x_0 + \sqrt{1-\overline{\alpha_t}} \cdot \epsilon_0",
            font_size=100
        ).shift(UP*2)
        
        arrow = Arrow(eq_14.get_bottom(), eq_14.get_bottom() + DOWN, color=YELLOW, buff=0.1, stroke_width=6)
        
        eq_93 = MathTex(
            r"\Rightarrow x_0 = \frac{x_t - \sqrt{1-\overline{\alpha_t}}\epsilon_0}{\sqrt{\overline{\alpha_t}}}",
            font_size=100
        ).next_to(arrow, DOWN, buff=0.3)
        
        self.play(Write(desc))
        self.play(Write(eq_14))
        self.wait(1)
        self.play(GrowArrow(arrow))
        self.play(Write(eq_93))
        self.wait(3)
        self.play(FadeOut(desc), FadeOut(eq_14), FadeOut(arrow), FadeOut(eq_93))
    
    def show_mean_derivation(self):
        desc = Text("Substituting x_0 into Mean Formula", font_size=30, color=GREEN)
        desc.to_edge(DOWN * 3)

        eq_94 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t) x_0}{1-\overline{\alpha_t}}",
            font_size=60
        )
        eq_94_x0 = SurroundingRectangle(eq_94[0][39:41], color=RED)
        
        eq_95 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t)\frac{x_t - \sqrt{1-\overline{\alpha_t}}\epsilon_0}{\sqrt{\overline{\alpha_t}}}}{1-\overline{\alpha_t}}",
            font_size=50
        )

        eq_96 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t)\frac{x_t - \sqrt{1-\overline{\alpha_t}}\epsilon_0}{\sqrt{\alpha_t} \sqrt{\overline{\alpha_{t-1}}}}}{1-\overline{\alpha_t}}",
            font_size=50
        )

        # box1 = SurroundingRectangle(eq_73[0][45:58], color=RED, buff=0.05)  # denominator of first fraction
        alpha_t_1_box1 = SurroundingRectangle(eq_96[0][27:33], color=RED)
        alpha_t_1_box2 = SurroundingRectangle(eq_96[0][56:59], color=RED)

        eq_97 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + (1-\alpha_t)(x_t - \sqrt{1-\overline{\alpha_t}}\epsilon_0)}{\sqrt{\alpha_t}(1-\overline{\alpha_t})}",
            font_size=50
        )
        
        self.play(Write(desc))
        self.play(Write(eq_94))
        self.wait(1)
        self.play(Write(eq_94_x0))
        self.wait(1)
        self.play(FadeOut(eq_94_x0))

        self.play(ReplacementTransform(eq_94, eq_95))
        self.wait(3)

        self.play(ReplacementTransform(eq_95, eq_96))
        self.wait(2)

        self.play(Write(alpha_t_1_box1))
        self.play(Write(alpha_t_1_box2))
        self.wait(2)
        self.play(FadeOut(alpha_t_1_box1), FadeOut(alpha_t_1_box2))

        self.play(ReplacementTransform(eq_96, eq_97))
        self.wait(3)

        self.play(FadeOut(desc))

        return eq_97
  
    def show_expanding_and_separating(self, eq_97):
        desc = Text("Expanding and Separating x_t and ε_0 Terms", font_size=30, color=GREEN)
        desc.to_edge(DOWN)
        
        # Equation 98 - Expanded form
        eq_98_1 = MathTex(
            r"\mu_q(x_t, x_0) = ",
            font_size=100
        )
        eq_98_2 = MathTex(
            r"\left( \alpha_t x_t - \alpha_t\overline{\alpha_{t-1}}x_t + x_t ",
            font_size=100
        )
        eq_98_3 = MathTex(
            r"- \sqrt{1-\overline{\alpha_t}} \epsilon_0 - \alpha_t x_t + \alpha_t \sqrt{1-\overline{\alpha_t}} \epsilon_0 \right)", 
            font_size=80
        )
        eq_98_4 = MathTex(
            r"/ \sqrt{\alpha_t}(1-\overline{\alpha_t})",
            font_size=100
        )
        # eq_98_3 = MathTex(
        #     r"\mu_q(x_t, x_0) = \frac{\alpha_t x_t - \alpha_t\overline{\alpha_{t-1}}x_t + x_t - \sqrt{1-\overline{\alpha_t}} \epsilon_0 - \alpha_t x_t + \alpha_t \sqrt{1-\overline{\alpha_t}} \epsilon_0}{\sqrt{\alpha_t}(1-\overline{\alpha_t})}",
        #     font_size=26
        # )
        eq_98 = VGroup(eq_98_1, eq_98_2, eq_98_3, eq_98_4).arrange(DOWN, buff=0.5)
        
        self.play(Write(desc))
        # self.play(Write(eq_98))
        self.play(ReplacementTransform(eq_97, eq_98))
        self.wait(2)
        self.play(FadeOut(desc))

        # Visually highlight the canceling α_t·x_t terms
        # First, identify and box the terms that will cancel
        # Note: Adjust indices based on actual MathTex parsing
        cancel_note = Text("Watch the α_t·x_t terms", font_size=28, color=YELLOW)
        cancel_note.next_to(eq_98, DOWN, buff=0.5)
        
        self.play(FadeIn(cancel_note, shift=UP))
        self.wait(1)
        
        # Create boxes around the canceling terms
        # Positive α_t·x_t (approximate position in numerator)
        box1 = SurroundingRectangle(eq_98_2[0][1:5], color=RED, buff=0.05)
        label1 = MathTex(r"+\alpha_tx_t", font_size=32, color=RED)
        label1.next_to(box1, UP, buff=0.2)
        
        # Negative α_t·x_t (approximate position in numerator)
        box2 = SurroundingRectangle(eq_98_3[0][11:15], color=BLUE, buff=0.05)
        label2 = MathTex(r"-\alpha_tx_t", font_size=32, color=BLUE)
        label2.next_to(box2, DOWN, buff=0.2)
        
        self.play(Create(box1), FadeIn(label1, shift=DOWN))
        self.wait(1)
        self.play(Create(box2), FadeIn(label2, shift=UP))
        self.wait(1)
        
        # Show cancellation with cross marks
        cross1 = Line(box1.get_corner(DL), box1.get_corner(UR), color=RED, stroke_width=4)
        cross2 = Line(box2.get_corner(DL), box2.get_corner(UR), color=BLUE, stroke_width=4)
        
        cancel_text = Text("These cancel!", font_size=28, color=GREEN)
        cancel_text.next_to(cancel_note, DOWN, buff=0.3)
        
        self.play(Create(cross1), Create(cross2))
        self.play(FadeIn(cancel_text, shift=UP))
        self.wait(2)
        
        self.play(
            FadeOut(box1), FadeOut(label1), FadeOut(cross1),
            FadeOut(box2), FadeOut(label2), FadeOut(cross2),
            FadeOut(cancel_note), FadeOut(cancel_text)
        )
        
        # Equation 99 - Separated into x_t and ε_0 terms
        desc2 = Text("Separating into x_t and ε_0 Coefficients", font_size=30, color=YELLOW)
        desc2.to_edge(DOWN)
        
        self.play(Write(desc2))
        
        eq_99_1 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{1-\overline{\alpha_t}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})}x_t ",
            font_size=100
        )
        eq_99_2 = MathTex(
            r"+ \frac{-\sqrt{1-\overline{\alpha_t}} + \alpha_t\sqrt{1-\overline{\alpha_t}}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})} \epsilon_0",
            font_size=100
        )
        eq_99 = VGroup(eq_99_1, eq_99_2).arrange(DOWN, buff=0.5)
        # eq_99_2 = MathTex(
        #     r"\mu_q(x_t, x_0) = \frac{1-\overline{\alpha_t}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})}x_t + \frac{-\sqrt{1-\overline{\alpha_t}} + \alpha_t\sqrt{1-\overline{\alpha_t}}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})} \epsilon_0",
        #     font_size=50
        # )
        eq_99.to_edge(UP)
        
        self.play(TransformMatchingTex(eq_98, eq_99))
        self.wait(3)
        self.play(FadeOut(desc2))

        return eq_99
        
    def show_cancellation_steps(self, eq_99):
        desc = Text("Canceling Common Terms", font_size=30, color=GREEN)
        desc.to_edge(DOWN * 3)
        
        # Equation 100 - Before cancellation, showing what will cancel
        # eq_100 = MathTex(
        #     r"\mu_q(x_t, x_0) = \frac{1-\overline{\alpha_t}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})}x_t - \frac{(1-\alpha_t)\sqrt{1-\overline{\alpha_t}}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})} \epsilon_0",
        #     font_size=80
        # )
        eq_100_1 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{1-\overline{\alpha_t}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})}x_t ",
            font_size=80
        )
        eq_100_2 = MathTex(
            r"- \frac{(1-\alpha_t)\sqrt{1-\overline{\alpha_t}}}{\sqrt{\alpha_t}(1-\overline{\alpha_t})} \epsilon_0",
            font_size=80
        )
        eq_100 = VGroup(eq_100_1, eq_100_2).arrange(DOWN, buff=0.3).shift(UP*1.5)
        
        self.play(Write(desc))
        # self.play(Write(eq_100))
        self.play(TransformMatchingTex(eq_99, eq_100))
        self.wait(1)
        
        # Highlight cancellations
        box1 = SurroundingRectangle(eq_100_1[0][10:15], color=RED, buff=0.05)
        box2 = SurroundingRectangle(eq_100_1[0][20:27], color=RED, buff=0.05)
        # cancel_x = Text("In x_t coefficient: (1-ᾱ_t) cancels!", font_size=24, color=BLUE)
        # cancel_x.next_to(eq_100, DOWN, buff=0.3)
        
        # cancel_eps = Text("In ε_0 coefficient: √(1-ᾱ_t) cancels!", font_size=24, color=GREEN)
        # cancel_eps.next_to(cancel_x, DOWN, buff=0.2)
        
        self.play(Write(box1), Write(box2))
        # self.wait(1)
        # self.play(FadeIn(cancel_eps, shift=UP))
        self.wait(2)
        
        self.play(FadeOut(box1), FadeOut(box2))
        
        # Equation 101 - After cancellation
        desc2 = Text("After Cancellation", font_size=36, color=YELLOW)
        desc2.to_edge(3 * DOWN)
        
        self.play(ReplacementTransform(desc, desc2))
        
        eq_101 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\overline{\alpha_t}}}\epsilon_0",
            font_size=80
        )
        
        self.play(TransformMatchingTex(eq_100, eq_101))
        # self.play(ReplacementTransform(var_calc1, var_calc2))
        self.wait(2)
        self.play(FadeOut(desc2))

        return eq_101
    
    def show_final_mean_form(self, eq_101):
        desc = Text("Factoring Out 1/√α_t", font_size=30, color=GREEN)
        desc.to_edge(UP)
        
        # Equation 102 - Final form
        eq_102 = MathTex(
            r"\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_0\right)",
            font_size=80,
            color=GOLD
        )
        
        self.play(Write(desc))
        self.play(ReplacementTransform(eq_101, eq_102))
        # self.play(Write(eq_102))
        self.wait(3)
        
        ## Highlight the structure
        #box1 = SurroundingRectangle(eq_102[0][0:5], color=BLUE, buff=0.05)
        #label1 = Text("Scale factor", font_size=22, color=BLUE).next_to(box1, LEFT, buff=0.3)
        
        #box2 = SurroundingRectangle(eq_102[0][6:8], color=GREEN, buff=0.05)
        #label2 = Text("Current noisy image", font_size=22, color=GREEN).next_to(box2, DOWN, buff=0.3)
        
        #box3 = SurroundingRectangle(eq_102[0][9:18], color=PURPLE, buff=0.05)
        #label3 = Text("Predicted noise (scaled)", font_size=22, color=PURPLE).next_to(box3, RIGHT, buff=0.3)
        
        #self.play(Create(box1), FadeIn(label1))
        #self.wait(1)
        #self.play(Create(box2), FadeIn(label2))
        #self.wait(1)
        #self.play(Create(box3), FadeIn(label3))
        #self.wait(2)
        
        #self.play(
        #    FadeOut(box1), FadeOut(label1),
        #    FadeOut(box2), FadeOut(label2),
        #    FadeOut(box3), FadeOut(label3)
        #)
        
        # Key insight
        note = Text(
            "This is the 'perfect' denoising mean we want our model to learn!",
            font_size=28,
            color=GREEN
        ).shift(DOWN*1.5)
        
        model_eq = MathTex(
            r"\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_\theta(x_t, t)\right)",
            font_size=70,
            color=RED
        )
        
        model_note = Text("Our model predicts ε_θ(x_t, t) ≈ ε_0", font_size=24, color=RED)
        model_note.next_to(model_eq, DOWN, buff=0.3)
        
        self.play(FadeIn(note, shift=UP))
        self.wait(2)
        # self.play(Write(model_eq))
        self.play(ReplacementTransform(eq_102, model_eq))
        # self.play(TransformMatchingTex(eq_102, model_eq))
        self.play(FadeOut(note))
        self.play(FadeIn(model_note, shift=UP))
        self.wait(3)
        
        self.play(FadeOut(desc), 
                  FadeOut(model_eq), FadeOut(model_note))
  
    def show_sampling_algorithm(self):
        title = Text("DDPM Sampling Algorithm", font_size=44, color=GOLD)
        title.to_edge(UP)

        algo = ImageMobject("sampling_algo.png")

        # Optionally, adjust the size or position of the image
        algo.scale(0.8)  # Scale down the image
        # logo.to_edge(UP + LEFT) # Move it to the top-left corner
        self.add(algo)
        self.play(FadeIn(algo))
        
        # Create algorithm box
        # algo_code = Code(
            # code_string="""
# 1: x_T ~ N(0, I)
#2: for t = T, ..., 1 do
#3:   z ~ N(0, I) if t > 1, else z = 0
#4:   x_{t-1} = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t, t)) + σ_t·z
#5: end for
#6: return x_0
#            """.strip(),
        #     language="python",
        #     # font="Monospace",
        #     # font_size=24,
        #     background="window",
        #     add_line_numbers =False,
        #     formatter_style="monokai"
        # ).scale(0.8)
        
        self.play(Write(title))
        # self.play(FadeIn(algo_code, shift=UP))
        self.wait(3)
        self.play(FadeOut(title), FadeOut(algo))
    
    def show_algorithm_walkthrough(self):
        title = Text("Algorithm Walkthrough", font_size=40, color=YELLOW)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Step 1
        step1 = VGroup(
            Text("Step 1: Initialize with pure noise", font_size=32, color=BLUE),
            MathTex(r"x_T \sim \mathcal{N}(0, I)", font_size=36)
        ).arrange(DOWN, buff=0.3).shift(UP*1.5)
        
        self.play(FadeIn(step1, shift=UP))
        self.wait(2)
        self.play(FadeOut(step1))
        
        # Step 2
        step2 = VGroup(
            Text("Step 2: Loop backwards from T to 1", font_size=32, color=GREEN),
            Text("At each timestep, predict and remove noise", font_size=28, color=WHITE)
        ).arrange(DOWN, buff=0.3).shift(UP*1)
        
        self.play(FadeIn(step2, shift=UP))
        self.wait(2)
        self.play(FadeOut(step2))
        
        # Step 3
        step3 = VGroup(
            Text("Step 3: Sample noise z (except at t=1)", font_size=32, color=PURPLE),
            MathTex(r"z \sim \mathcal{N}(0, I) \text{ if } t > 1, \text{ else } z = 0", font_size=32)
        ).arrange(DOWN, buff=0.3).shift(UP*1)
        
        self.play(FadeIn(step3, shift=UP))
        self.wait(2)
        self.play(FadeOut(step3))
        
        # Step 4 - The core denoising step
        step4_title = Text("Step 4: Denoising Step (The Heart of DDPM!)", font_size=32, color=RED)
        step4_title.shift(UP*2)
        
        denoising_eq = MathTex(
            r"x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_\theta(x_t, t)\right) + \sigma_t z",
            font_size=70
        ).shift(UP*0.5)
        
        # Break down the equation
        part1 = Text("Scale factor", font_size=24, color=BLUE).shift(LEFT*4 + DOWN*0.5)
        arrow1 = Arrow(part1.get_top(), denoising_eq[0][5:10].get_bottom(), color=BLUE, buff=0.1)
        
        part2 = Text("Predicted noise", font_size=24, color=GREEN).shift(RIGHT*3 + DOWN*0.8)
        arrow2 = Arrow(part2.get_top(), denoising_eq[0][30:31].get_bottom(), color=GREEN, buff=0.1)
        
        part3 = Text("Random noise", font_size=24, color=PURPLE).shift(RIGHT*4 + DOWN*1.8)
        arrow3 = Arrow(part3.get_top(), denoising_eq[0][-2:].get_bottom(), color=PURPLE, buff=0.1)
        
        self.play(Write(step4_title))
        self.play(Write(denoising_eq))
        self.wait(1)
        self.play(
            FadeIn(part1, shift=UP), GrowArrow(arrow1),
            FadeIn(part2, shift=UP), GrowArrow(arrow2),
            FadeIn(part3, shift=UP), GrowArrow(arrow3)
        )
        self.wait(3)
        self.play(
            FadeOut(step4_title), FadeOut(denoising_eq),
            FadeOut(part1), FadeOut(arrow1),
            FadeOut(part2), FadeOut(arrow2),
            FadeOut(part3), FadeOut(arrow3)
        )
        
        # Final step
        step5 = VGroup(
            Text("Step 5: Return denoised image x_0", font_size=32, color=GOLD),
            Text("After T iterations, we've recovered our image!", font_size=28, color=WHITE)
        ).arrange(DOWN, buff=0.3)
        
        self.play(FadeIn(step5, shift=UP))
        self.wait(3)
        self.play(FadeOut(title), FadeOut(step5))


# To render:
# python -m manim -pql ddpm.py DDPMGaussianSampling

###########################################

from manim import *

class DDPMTrainingAlgorithm(Scene):
    def construct(self):
        self.camera.background_color = "#0f0f23"

        self.my_template = TexTemplate()
        #self.my_template = TexTemplate(
        #    documentclass="\\documentclass[preview]{standalone}",
        #    preamble="""
        #    \\usepackage{amsmath}
        #    \\DeclareMathOperator{\\Res}{Res}
        #    \\DeclareMathOperator{\\End}{End}
        #    """
        #)
        
        # Title
        self.show_title()
        
        # Part 1: KL Divergence Foundation
        self.show_kl_divergence_setup()
        
        # Part 2: Applying to our problem
        self.show_applying_kl_to_loss()
        self.show_kl_divergence_general()
        
        # Part 3: Simplifying the loss
        self.show_kl_expansion()
        #self.show_canceling_terms()
        #self.show_variance_substitution()
        
        ## Part 4: Substituting means
        #self.show_substituting_means()
        #self.show_expanding_means()
        #self.show_canceling_xt_terms()
        
        ## Part 5: Factoring and final form
        #self.show_factoring_epsilon_terms()
        #self.show_coefficient_squared()
        #self.show_final_loss_form()
        
        ## Part 6: Training algorithm
        #self.show_training_algorithm()
        
    def show_title(self):
        title = Text("DDPM Training Algorithm", font_size=48, color=BLUE)
        subtitle = Text("Deriving the Loss Function", font_size=32, color=PURPLE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

    def show_kl_divergence_general(self):
        desc = Text("KL Divergence for Multivariate Gaussians", font_size=40, color=YELLOW)
        desc.to_edge(UP)
        
        # General KL divergence formula
        kl_general = MathTex(
            r"D_{KL}(\mathcal{N}(x; \mu_x, \Sigma_x) \parallel \mathcal{N}(y; \mu_y, \Sigma_y)) =",
            font_size=60
        ).shift(UP*2)
        
        kl_terms = VGroup(
            # MathTex(r"\frac{1}{2}\left[\log\frac{\Sigma_y}{\Sigma_x} - d + tr(\Sigma_y^{-1}\Sigma_x) + (\mu_y - \mu_x)^T\Sigma_y^{-1}(\mu_y - \mu_x)\right]", font_size=30, color=GREEN)
            MathTex(r"\frac{1}{2}\left[\log\frac{\Sigma_y}{\Sigma_x} - d", font_size=60), 
            MathTex(r" + tr(\Sigma_y^{-1}\Sigma_x) + ", font_size=60),
            MathTex(r"\mu_y - \mu_x)^T\Sigma_y^{-1}(\mu_y - \mu_x)\right]", font_size=60)
        ).arrange(DOWN).shift(DOWN*1)
        
        kl_combined = VGroup(kl_general, kl_terms)
        
        self.play(Write(desc))
        self.play(Write(kl_general))
        self.play(Write(kl_terms))
        self.wait(3)
        self.play(FadeOut(desc), FadeOut(kl_combined))
    
    def show_kl_divergence_setup(self):
        desc = Text("For Our Problem: Two Gaussians with Same Dimension d", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        setup = VGroup(
            MathTex(r"\text{Ground Truth: } q(x_{t-1}|x_t, x_0) \sim \mathcal{N}(\mu_q, \Sigma_q(t))", font_size=60, color=GREEN),
            MathTex(r"\text{Model: } p_\theta(x_{t-1}|x_t) \sim \mathcal{N}(\mu_\theta, \Sigma_q(t))", font_size=60, color=BLUE),
            MathTex(r"\text{Same Variance: } \Sigma_q(t) = \sigma_q^2(t)I", font_size=50, color=PURPLE)
        ).arrange(DOWN, buff=0.5)
        
        self.play(Write(desc))
        self.play(FadeIn(setup, shift=UP, lag_ratio=0.3))
        self.wait(3)
        self.play(FadeOut(desc), FadeOut(setup))
    
    def show_applying_kl_to_loss(self):
        desc = Text("From Denoising Loss Term (Eq. 73)", font_size=36, color=YELLOW)
        desc.to_edge(UP)

        loss_term = MathTex(
            r"\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p_\theta(x_{t-1}|x_t))",
            font_size=60,
            color=ORANGE
        ).shift(UP*1)
        
        # Equivalent form
        #equiv = MathTex(
        #    r"= \argmin_\theta D_{KL}(\mathcal{N}(x_{t-1}; \mu_q, \Sigma_q(t)) \parallel \mathcal{N}(x_{t-1}; \mu_\theta, \Sigma_q(t)))",
        #    font_size=36
        #).shift(DOWN*1)
        equiv = MathTex(
            r"= arg\min_\theta D_{KL}(",
            r"\mathcal{N}(x_{t-1}; \mu_q, \Sigma_q(t))",
            r"\parallel", 
            r"\mathcal{N}(x_{t-1}; \mu_\theta, \Sigma_q(t)))",
            font_size=60
        ).arrange_submobjects(RIGHT, buff=0.1).shift(DOWN*1)
        
        self.play(Write(desc))
        self.play(Write(loss_term))
        self.wait(2)
        self.play(Write(equiv))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(loss_term), FadeOut(equiv))
    
    def show_variance_setup(self):
        desc = Text("Variance Setup: Both Use Σ_q(t)", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        note = VGroup(
            Text("The authors fixed the variance at Σ_q(t) = σ²_q(t)I", font_size=28, color=WHITE),
            Text("Neural network only learns the mean μ_θ", font_size=28, color=GREEN),
            Text("This simplifies optimization significantly", font_size=28, color=GOLD)
        ).arrange(DOWN, buff=0.4).shift(UP*0.5)
        
        self.play(Write(desc))
        self.play(FadeIn(note, shift=UP, lag_ratio=0.2))
        self.wait(3)
        self.play(FadeOut(desc), FadeOut(note))
    
    def show_kl_expansion(self):
        desc = Text("Expanding KL Divergence", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        kl_expanded = MathTex(
            r"= \argmin_\theta \frac{1}{2}\left[\begin{array}{c}",
            r"\log\frac{\Sigma_q(t)}{\Sigma_q(t)} - d + tr(\Sigma_q(t)^{-1}\Sigma_q(t)) \\",
            r"+ (\mu_\theta - \mu_q)^T\Sigma_q(t)^{-1}(\mu_\theta - \mu_q)",
            r"\end{array}\right]",
            font_size=32
        )
        
        self.play(Write(desc))
        self.play(Write(kl_expanded))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(kl_expanded))
    
    def show_canceling_terms(self):
        desc = Text("Canceling Terms", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        # Show which terms cancel
        notes = VGroup(
            MathTex(r"\log\frac{\Sigma_q(t)}{\Sigma_q(t)} = \log 1 = 0 \quad \text{(cancels)}", font_size=32, color=RED),
            MathTex(r"tr(\Sigma_q(t)^{-1}\Sigma_q(t)) = tr(I) = d \quad \text{(constant)}", font_size=32, color=BLUE),
            MathTex(r"-d + d = 0 \quad \text{(cancel each other)}", font_size=32, color=GREEN)
        ).arrange(DOWN, buff=0.5).shift(UP*0.5)
        
        result = MathTex(
            r"\Rightarrow \argmin_\theta \frac{1}{2}(\mu_\theta - \mu_q)^T\Sigma_q(t)^{-1}(\mu_\theta - \mu_q)",
            font_size=36,
            color=GOLD
        ).shift(DOWN*1.5)
        
        self.play(Write(desc))
        self.play(FadeIn(notes, shift=UP, lag_ratio=0.2))
        self.wait(2)
        self.play(Write(result))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(notes), FadeOut(result))
    
    def show_variance_substitution(self):
        desc = Text("Substituting Σ_q(t) = σ²_q(t)I", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        before = MathTex(
            r"= \argmin_\theta \frac{1}{2}(\mu_\theta - \mu_q)^T(\sigma_q^2(t)I)^{-1}(\mu_\theta - \mu_q)",
            font_size=34
        ).shift(UP*1)
        
        after = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)}[(\mu_\theta - \mu_q)^T(\mu_\theta - \mu_q)]",
            font_size=34
        ).shift(DOWN*0.5)
        
        # Define L2 norm
        norm_def = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)}|||\mu_\theta - \mu_q||_2^2",
            font_size=34,
            color=GREEN
        ).shift(DOWN*1.5)
        
        self.play(Write(desc))
        self.play(Write(before))
        self.wait(1)
        self.play(Write(after))
        self.wait(1)
        self.play(Write(norm_def))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(before), FadeOut(after), FadeOut(norm_def))
    
    def show_substituting_means(self):
        desc = Text("Substituting the Means: μ_q and μ_θ", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        # Recall the means
        mu_q = MathTex(
            r"\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_0\right)",
            font_size=32,
            color=BLUE
        ).shift(UP*1.5)
        
        mu_theta = MathTex(
            r"\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)",
            font_size=32,
            color=GREEN
        ).shift(UP*0.2)
        
        self.play(Write(desc))
        self.play(Write(mu_q))
        self.wait(1)
        self.play(Write(mu_theta))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(mu_q), FadeOut(mu_theta))
    
    def show_expanding_means(self):
        desc = Text("Expanding Inside the Norm", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        expansion = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)}\left||",
            r"\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)",
            r"- \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_0\right)",
            r"\right||_2^2",
            font_size=28
        ).shift(UP*0.5)
        
        # Show simplification
        simplified = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)}\left||",
            r"\frac{x_t}{\sqrt{\alpha_t}} - \frac{(1-\alpha_t)}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)",
            r"- \frac{x_t}{\sqrt{\alpha_t}} + \frac{(1-\alpha_t)}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\epsilon_0",
            r"\right||_2^2",
            font_size=26
        ).shift(DOWN*1.2)
        
        self.play(Write(desc))
        self.play(Write(expansion))
        self.wait(2)
        self.play(Write(simplified))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(expansion), FadeOut(simplified))
    
    def show_canceling_xt_terms(self):
        desc = Text("Canceling x_t Terms", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        note = MathTex(
            r"\frac{x_t}{\sqrt{\alpha_t}} - \frac{x_t}{\sqrt{\alpha_t}} = 0 \quad \text{(cancels!)}",
            font_size=40,
            color=RED
        ).shift(UP*1)
        
        result = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)}\left||",
            r"\frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}(\epsilon_0 - \epsilon_\theta(x_t, t))",
            r"\right||_2^2",
            font_size=32,
            color=GREEN
        ).shift(DOWN*1)
        
        self.play(Write(desc))
        self.play(Write(note))
        self.wait(2)
        self.play(Write(result))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(note), FadeOut(result))
    
    def show_factoring_epsilon_terms(self):
        desc = Text("Factoring Out the Coefficient", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        before = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)}\left||",
            r"\frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}(\epsilon_0 - \epsilon_\theta(x_t, t))",
            r"\right||_2^2",
            font_size=32
        ).shift(UP*1)
        
        # Extract the coefficient
        coeff = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)} \cdot ",
            r"\left(\frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\right)^2",
            r"||(\epsilon_0 - \epsilon_\theta(x_t, t))||_2^2",
            font_size=28
        ).shift(DOWN*1)
        
        self.play(Write(desc))
        self.play(Write(before))
        self.wait(1)
        self.play(Write(coeff))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(before), FadeOut(coeff))
    
    def show_coefficient_squared(self):
        desc = Text("Squaring the Coefficient", font_size=36, color=YELLOW)
        desc.to_edge(UP)
        
        # Show the squaring
        coeff_squared = MathTex(
            r"\left(\frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\right)^2 = ",
            r"\frac{(1-\alpha_t)^2}{\alpha_t(1-\bar{\alpha}_t)}",
            font_size=38,
            color=BLUE
        ).shift(UP*1.5)
        
        # Final form
        final_loss = MathTex(
            r"= \argmin_\theta \frac{(1-\alpha_t)^2}{2\sigma_q^2(t)\alpha_t(1-\bar{\alpha}_t)} ||(\epsilon_0 - \epsilon_\theta(x_t, t))||_2^2",
            font_size=32,
            color=GOLD
        ).shift(DOWN*1)
        
        # Replace with β_t
        beta_note = MathTex(
            r"\text{Note: } \beta_t = \frac{(1-\alpha_t)^2}{\alpha_t(1-\bar{\alpha}_t)} \text{ (from Eq. 7)}",
            font_size=28,
            color=GREEN
        ).shift(DOWN*2.2)
        
        self.play(Write(desc))
        self.play(Write(coeff_squared))
        self.wait(1)
        self.play(Write(final_loss))
        self.wait(1)
        self.play(FadeIn(beta_note, shift=UP))
        self.wait(2)
        self.play(FadeOut(desc), FadeOut(coeff_squared), FadeOut(final_loss), FadeOut(beta_note))
    
    def show_final_loss_form(self):
        desc = Text("Final Loss Function Form", font_size=40, color=YELLOW)
        desc.to_edge(UP)
        
        # Using beta_t
        loss_with_beta = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)} \frac{\beta_t^2}{||(\epsilon_0 - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_0, t))||_2^2}",
            font_size=28
        ).shift(UP*0.5)
        
        # Substitute x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * epsilon_0
        substitute = MathTex(
            r"\text{From Eq. 14: } x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_0",
            font_size=28,
            color=BLUE
        ).shift(DOWN*1)
        
        # Final clean form
        final = MathTex(
            r"= \argmin_\theta \frac{1}{2\sigma_q^2(t)} \frac{\beta_t^2}{||(\epsilon_0 - \epsilon_\theta(x_t, t))||_2^2}",
            font_size=40,
            color=GOLD
        ).shift(DOWN*2.2)
        
        highlight = SurroundingRectangle(final, color=GOLD, buff=0.15)
        
        self.play(Write(desc))
        self.play(Write(loss_with_beta))
        self.wait(1)
        self.play(FadeIn(substitute, shift=UP))
        self.wait(1)
        self.play(Write(final), Create(highlight))
        self.wait(3)
        self.play(FadeOut(desc), FadeOut(loss_with_beta), FadeOut(substitute), 
                  FadeOut(final), FadeOut(highlight))
    
    def show_training_algorithm(self):
        title = Text("Algorithm 1: DDPM Training", font_size=44, color=GOLD)
        title.to_edge(UP)
        
        algo = VGroup(
            MathTex(r"\textbf{repeat}", font_size=32),
            MathTex(r"\quad \mathbf{x_0} \sim q(\mathbf{x_0})", font_size=32),
            MathTex(r"\quad t \sim \text{Uniform}(\{1, \ldots, T\})", font_size=32),
            MathTex(r"\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)", font_size=32),
            MathTex(r"\quad \textbf{Take gradient descent step on}", font_size=32),
            MathTex(r"\qquad \nabla_\theta ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x_0} + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t)||^2", font_size=28, color=RED),
            MathTex(r"\textbf{until converged}", font_size=32),
        ).arrange(DOWN, buff=0.4).shift(DOWN*0.5)
        
        self.play(Write(title))
        self.play(FadeIn(algo, shift=UP, lag_ratio=0.15))
        self.wait(4)
        
        # Breakdown
        breakdown = VGroup(
            Text("1. Sample clean image x_0", font_size=28, color=BLUE),
            Text("2. Sample random timestep t", font_size=28, color=GREEN),
            Text("3. Sample random noise ε", font_size=28, color=PURPLE),
            Text("4. Predict noise and minimize difference", font_size=28, color=ORANGE),
            Text("5. Repeat until convergence", font_size=28, color=GOLD)
        ).arrange(DOWN, buff=0.3).shift(DOWN*3.5)
        
        self.play(FadeIn(breakdown, shift=UP, lag_ratio=0.1))
        self.wait(3)
        
        self.play(FadeOut(title), FadeOut(algo), FadeOut(breakdown))


# NARRATION
"""
[INTRO - KL Divergence Foundation]
To train our diffusion model, we need to minimize the KL divergence between 
the ground truth posterior and our learned model.
For multivariate Gaussians, the KL divergence has a specific form.

[KL Divergence General]
The KL divergence between two Gaussians has four terms:
log ratio of covariances, dimension d, trace of covariance products,
and a mean difference quadratic form.

[Setup]
In our case, we have two Gaussians with the same dimension:
the ground truth posterior q(x_{t-1}|x_t, x_0) and our model p_θ(x_{t-1}|x_t).
The key insight: both share the same variance Σ_q(t) = σ²_q(t)I!

[Applying KL]
Our denoising loss wants to minimize this KL divergence.
We can rewrite it as an argmin over our model parameters θ.

[Variance Setup]
Here's the crucial design choice: the authors fixed the variance!
The neural network only learns the mean μ_θ.
This dramatically simplifies the optimization problem.

[KL Expansion]
Expanding the KL divergence gives us four terms.

[Canceling Terms]
Three terms cancel out beautifully:
- The log ratio of identical covariances is log 1 = 0
- The trace terms give d, which cancels with -d
- We're left with just the mean difference term!

[Variance Substitution]
Substituting the specific variance form, we get a scaled L2 norm
between the two means.

[Substituting Means]
Now we substitute our derived expressions for μ_q and μ_θ.
μ_q comes from the ground truth denoising step.
μ_θ is what our network learns to predict.

[Expanding Means]
When we expand both inside the norm, something beautiful happens.

[Canceling x_t]
The x_t/√α_t terms appear in both means with opposite signs!
They completely cancel out, leaving only the noise terms.

[Factoring Epsilon]
We can factor out the coefficient that multiplies both epsilon terms.

[Coefficient Squared]
When we take it outside the norm, the coefficient gets squared.

[Final Loss Form]
Here's our final loss function!
The neural network ε_θ predicts the source noise ε_0.
We minimize the L2 distance between true and predicted noise.

[Training Algorithm]
And this leads to Algorithm 1: the training procedure!
Repeat: sample x_0, sample timestep t, sample noise ε,
take a gradient step on the noise prediction error.
Continue until convergence.

[Key Insight]
The elegance of this approach: we're not predicting images directly!
We're predicting the noise that was added.
This is much easier for the neural network to learn.
The training loop becomes simple: add noise to images, train network to predict it back.
"""

# To render:
# manim -pqh ddpm_training_algorithm.py DDPMTrainingAlgorithm