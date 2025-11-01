from manim import *
import numpy as np


class MCMCDiffusionVideo(Scene):
    def construct(self):
        # Scene 1: Opening [0:00-0:15]
        self.opening()
        self.wait(0.5)
        
        # Scene 2: Problem [0:15-0:45]
        self.problem()
        self.wait(0.5)
        
        # Scene 3: Key Insight [0:45-1:10]
        self.key_insight()
        self.wait(0.5)
        
        # Scene 4: Introducing MCMC [1:10-1:40]
        self.introducing_mcmc()
        self.wait(0.5)
        
        # Scene 5: Markovian Property [1:40-2:10]
        self.markovian_property()
        self.wait(0.5)
        
        # Scene 6: Markov Chain Structure [2:10-2:40]
        self.markov_chain_structure()
        self.wait(0.5)
        
        # Scene 7: Transition Matrix [2:40-3:10]
        self.transition_matrix()
        self.wait(0.5)
        
        # Scene 8: Convergence [3:10-3:50]
        self.convergence()
        self.wait(0.5)
        
        # Scene 9: Connection to Diffusion [3:50-4:20]
        self.connection_to_diffusion()
        self.wait(0.5)
        
        # Scene 10: Power of Indirection [4:20-4:50]
        self.power_of_indirection()
        self.wait(0.5)
    
    def opening(self):
        # Title
        title = Text("MCMC for Diffusion Models", font_size=60, gradient=(BLUE, PURPLE))
        subtitle = Text("Sampling from Unknown Distributions", font_size=36, color=GRAY)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
    
    def problem(self):
        # Title
        title = Text("The Problem", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Goal statement
        goal = Text("Goal: Sample from the distribution of real images", font_size=32)
        goal.next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(goal))
        self.wait(1)
        
        # Show image grid
        image_grid = VGroup()
        for i in range(4):
            for j in range(4):
                square = Square(side_length=0.6, fill_opacity=0.3, fill_color=random_bright_color())
                square.move_to(RIGHT * (j - 1.5) * 0.7 + DOWN * (i - 1) * 0.7)
                image_grid.add(square)
        
        self.play(Create(image_grid), run_time=2)
        self.wait(1)
        
        # Show the challenge
        challenge = Text("Challenge: The true distribution is unknown!", font_size=36, color=RED)
        challenge.to_edge(DOWN)
        self.play(Write(challenge))
        
        # Add question marks
        questions = VGroup(*[Text("?", font_size=60, color=YELLOW).move_to(sq.get_center()) 
                            for sq in image_grid])
        self.play(FadeIn(questions, lag_ratio=0.1))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, goal, image_grid, questions, challenge)))
    
    def key_insight(self):
        # Question
        question = Text("Do we need to know the exact distribution?", font_size=40, color=YELLOW)
        question.to_edge(UP)
        self.play(Write(question))
        self.wait(1)
        
        # Answer
        answer = Text("No! We just need to SAMPLE from it!", font_size=48, color=GREEN)
        self.play(FadeIn(answer, shift=UP))
        self.wait(2)
        
        # Show distinction
        self.play(FadeOut(question), answer.animate.to_edge(UP))
        
        # Two approaches
        left_box = Rectangle(width=5, height=3, color=RED).shift(LEFT * 3.5)
        left_title = Text("Find Distribution", font_size=28, color=RED)
        left_title.next_to(left_box, UP)
        left_text = Text("Hard!\nComplex\nIntractable", font_size=24)
        left_text.move_to(left_box)
        
        right_box = Rectangle(width=5, height=3, color=GREEN).shift(RIGHT * 3.5)
        right_title = Text("Sample from It", font_size=28, color=GREEN)
        right_title.next_to(right_box, UP)
        right_text = Text("Possible!\nMCMC\nIterative", font_size=24)
        right_text.move_to(right_box)
        
        self.play(Create(left_box), Write(left_title), Write(left_text))
        self.wait(1)
        self.play(Create(right_box), Write(right_title), Write(right_text))
        self.wait(2)
        
        # Highlight solution
        self.play(Indicate(right_box, scale_factor=1.1, color=YELLOW), run_time=2)
        self.wait(1)
        
        self.play(FadeOut(VGroup(answer, left_box, left_title, left_text, 
                                 right_box, right_title, right_text)))
    
    def introducing_mcmc(self):
        # Title
        title = Text("Markov Chain Monte Carlo (MCMC)", font_size=48, gradient=(BLUE, PURPLE))
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Key idea
        idea = Text("Build a Markov chain that approximates the target distribution", 
                   font_size=32)
        idea.next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(idea))
        self.wait(2)
        
        # Show convergence concept
        convergence_text = Text("More steps → Better approximation", font_size=36, color=YELLOW)
        convergence_text.to_edge(DOWN)
        
        # Create simple chain visualization
        dots = VGroup()
        arrows = VGroup()
        for i in range(6):
            dot = Dot(radius=0.15, color=BLUE).shift(LEFT * 5 + RIGHT * i * 2)
            dots.add(dot)
            if i < 5:
                arrow = Arrow(dot.get_right(), dot.get_right() + RIGHT * 1.8, 
                             buff=0.2, color=WHITE)
                arrows.add(arrow)
        
        self.play(Create(dots), Create(arrows), run_time=2)
        self.play(Write(convergence_text))
        
        # Animate progression
        colors = [RED, ORANGE, YELLOW, GREEN_C, GREEN, BLUE]
        for i, (dot, color) in enumerate(zip(dots, colors)):
            self.play(dot.animate.set_color(color), run_time=0.3)
        
        self.wait(2)
        self.play(FadeOut(VGroup(title, idea, convergence_text, dots, arrows)))
    
    def markovian_property(self):
        # Title
        title = Text("The Markovian Property", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Definition
        definition = Text("Future depends ONLY on the present, not the past", 
                         font_size=36, color=YELLOW)
        definition.next_to(title, DOWN, buff=0.8)
        self.play(Write(definition))
        self.wait(2)
        
        # Visual representation
        past = Rectangle(width=2, height=1.5, color=GRAY, fill_opacity=0.3)
        past.shift(LEFT * 4)
        past_label = Text("Past", font_size=24, color=GRAY)
        past_label.next_to(past, UP)
        
        present = Rectangle(width=2, height=1.5, color=GREEN, fill_opacity=0.5)
        present_label = Text("Present", font_size=24, color=GREEN)
        present_label.next_to(present, UP)
        
        future = Rectangle(width=2, height=1.5, color=BLUE, fill_opacity=0.3)
        future.shift(RIGHT * 4)
        future_label = Text("Future", font_size=24, color=BLUE)
        future_label.next_to(future, UP)
        
        self.play(Create(past), Write(past_label),
                 Create(present), Write(present_label),
                 Create(future), Write(future_label))
        self.wait(1)
        
        # Show no connection from past
        no_arrow = Line(past.get_right(), future.get_left(), color=RED, stroke_width=8)
        cross = VGroup(
            Line(no_arrow.get_start() + UP * 0.3, no_arrow.get_end() + DOWN * 0.3, color=RED),
            Line(no_arrow.get_start() + DOWN * 0.3, no_arrow.get_end() + UP * 0.3, color=RED)
        )
        
        # Show connection from present
        yes_arrow = Arrow(present.get_right(), future.get_left(), color=GREEN, 
                         buff=0.2, stroke_width=8)
        checkmark = Text("✓", font_size=60, color=GREEN).move_to(yes_arrow)
        
        self.play(Create(no_arrow), Create(cross))
        self.wait(1)
        self.play(FadeOut(no_arrow), FadeOut(cross))
        self.play(Create(yes_arrow))
        self.play(Write(checkmark))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, definition, past, past_label, present, 
                                 present_label, future, future_label, yes_arrow, checkmark)))
    
    def markov_chain_structure(self):
        # Title
        title = Text("Markov Chain Structure", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create chain
        states = VGroup()
        labels = VGroup()
        for i in range(5):
            circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.3)
            circle.shift(LEFT * 4 + RIGHT * i * 2)
            states.add(circle)
            
            label = Text(f"S{i}", font_size=28)
            label.move_to(circle)
            labels.add(label)
        
        self.play(Create(states), Write(labels))
        self.wait(1)
        
        # Add transitions
        transitions = VGroup()
        for i in range(4):
            arrow = CurvedArrow(states[i].get_right() + RIGHT * 0.1, 
                               states[i+1].get_left() + LEFT * 0.1,
                               color=YELLOW)
            transitions.add(arrow)
            
            # Add probability label
            prob = MathTex(r"P_{" + str(i) + r"," + str(i+1) + r"}", font_size=24)
            prob.next_to(arrow, UP, buff=0.1)
            transitions.add(prob)
        
        self.play(Create(transitions), run_time=2)
        self.wait(1)
        
        # Add self-loops
        self_loops = VGroup()
        for i in [0, 2, 4]:
            loop = CurvedArrow(states[i].get_top() + UP * 0.1, 
                              states[i].get_top() + UP * 0.1 + RIGHT * 0.1,
                              angle=TAU/2, color=GREEN)
            self_loops.add(loop)
        
        self.play(Create(self_loops))
        
        explanation = Text("Each state can transition to others or stay put", 
                          font_size=28, color=GRAY)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, states, labels, transitions, self_loops, explanation)))
    
    def transition_matrix(self):
        # Title
        title = Text("Transition Probability Matrix", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create matrix
        matrix_values = [
            ["0.7", "0.2", "0.1"],
            ["0.3", "0.5", "0.2"],
            ["0.1", "0.3", "0.6"]
        ]
        
        matrix = Matrix(matrix_values, h_buff=1.2)
        matrix.scale(0.8)
        
        # Labels
        from_label = Text("From →", font_size=24, color=YELLOW)
        from_label.next_to(matrix, LEFT, buff=0.5)
        
        to_label = Text("↓ To", font_size=24, color=YELLOW)
        to_label.next_to(matrix, UP, buff=0.5)
        
        self.play(Create(matrix))
        self.play(Write(from_label), Write(to_label))
        self.wait(2)
        
        # Highlight row
        row_rect = Rectangle(width=matrix.width * 0.95, height=1, color=GREEN)
        row_rect.move_to(matrix.get_rows()[0])
        
        row_text = Text("Each row sums to 1", font_size=32, color=GREEN)
        row_text.to_edge(DOWN)
        
        self.play(Create(row_rect))
        self.play(Write(row_text))
        self.wait(2)
        
        # Show sum
        sum_text = MathTex("0.7 + 0.2 + 0.1 = 1.0", font_size=36, color=GREEN)
        sum_text.next_to(row_rect, RIGHT, buff=1)
        self.play(Write(sum_text))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, matrix, from_label, to_label, 
                                 row_rect, row_text, sum_text)))
    
    def convergence(self):
        # Title
        title = Text("Convergence to Target Distribution", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create two distributions
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1, 0.2],
            x_length=10,
            y_length=3,
            tips=False
        ).shift(DOWN * 0.5)
        
        # Target distribution (smooth curve)
        target = axes.plot(lambda x: 0.7 * np.exp(-0.5 * ((x - 5) / 1.5) ** 2), 
                          color=GREEN, stroke_width=4)
        target_label = Text("Target Distribution", font_size=24, color=GREEN)
        target_label.next_to(axes, DOWN, buff=0.3)
        
        self.play(Create(axes))
        self.play(Create(target), Write(target_label))
        self.wait(1)
        
        # Initial approximation (rough)
        current_points = [0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.3, 0.2, 0.2, 0.1]
        
        for step in range(5):
            bars = VGroup()
            for i, height in enumerate(current_points):
                bar = Rectangle(
                    width=0.8,
                    height=height * 3,
                    fill_opacity=0.7,
                    fill_color=BLUE,
                    color=BLUE
                ).align_to(axes.c2p(i, 0), DOWN).align_to(axes.c2p(i, 0), LEFT)
                bars.add(bar)
            
            step_label = Text(f"Step {step + 1}", font_size=32, color=YELLOW)
            step_label.to_edge(UP).shift(DOWN * 0.8)
            
            if step == 0:
                self.play(Create(bars), Write(step_label))
            else:
                self.play(ReplacementTransform(prev_bars, bars), ReplacementTransform(prev_label, step_label))
            
            prev_bars = bars
            prev_label = step_label
            self.wait(0.8)
            
            # Update distribution to converge
            target_func = lambda x: 0.7 * np.exp(-0.5 * ((x - 5) / 1.5) ** 2)
            current_points = [0.7 * target_func(i) + 0.3 * current_points[i] 
                             for i in range(len(current_points))]
        
        # Final message
        final_msg = Text("Approximation → Target", font_size=36, color=YELLOW)
        final_msg.to_edge(DOWN).shift(UP * 0.3)
        self.play(Write(final_msg))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, axes, target, target_label, 
                                 prev_bars, prev_label, final_msg)))
    
    def connection_to_diffusion(self):
        # Title
        title = Text("Connection to Diffusion Models", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show parallel concepts
        mcmc_box = Rectangle(width=5, height=4, color=BLUE).shift(LEFT * 3.5)
        mcmc_title = Text("MCMC", font_size=36, color=BLUE)
        mcmc_title.next_to(mcmc_box, UP)
        
        mcmc_points = VGroup(
            Text("• Iterative process", font_size=24),
            Text("• Gradual improvement", font_size=24),
            Text("• Converges to target", font_size=24),
            Text("• Samples distribution", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        mcmc_points.move_to(mcmc_box)
        
        diffusion_box = Rectangle(width=5, height=4, color=PURPLE).shift(RIGHT * 3.5)
        diffusion_title = Text("Diffusion", font_size=36, color=PURPLE)
        diffusion_title.next_to(diffusion_box, UP)
        
        diffusion_points = VGroup(
            Text("• Iterative denoising", font_size=24),
            Text("• Noise → Image", font_size=24),
            Text("• Converges to image", font_size=24),
            Text("• Generates samples", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        diffusion_points.move_to(diffusion_box)
        
        self.play(Create(mcmc_box), Write(mcmc_title))
        self.play(Write(mcmc_points), lag_ratio=0.3)
        self.wait(1)
        
        self.play(Create(diffusion_box), Write(diffusion_title))
        self.play(Write(diffusion_points), lag_ratio=0.3)
        self.wait(2)
        
        # Draw connections
        connections = VGroup()
        for i in range(4):
            arrow = Arrow(mcmc_points[i].get_right(), diffusion_points[i].get_left(),
                         color=YELLOW, buff=0.5, stroke_width=2)
            connections.add(arrow)
        
        self.play(Create(connections), run_time=2)
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, mcmc_box, mcmc_title, mcmc_points,
                                 diffusion_box, diffusion_title, diffusion_points, connections)))
    
    def power_of_indirection(self):
        # Title
        title = Text("The Power of Indirection", font_size=48, gradient=(BLUE, PURPLE))
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show the key insight
        insight1 = Text("We can't compute the distribution...", font_size=36, color=RED)
        insight1.shift(UP * 1.5)
        
        insight2 = Text("But we CAN build a process to sample from it!", 
                       font_size=36, color=GREEN)
        insight2.shift(DOWN * 0.5)
        
        self.play(Write(insight1))
        self.wait(1)
        self.play(Write(insight2))
        self.wait(2)
        
        # Visual metaphor
        metaphor = Text("Building a pathway instead of a map", 
                       font_size=32, color=YELLOW, slant=ITALIC)
        metaphor.to_edge(DOWN)
        self.play(FadeIn(metaphor, shift=UP))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, insight1, insight2, metaphor)))
    