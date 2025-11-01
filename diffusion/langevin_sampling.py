from manim import *
import numpy as np

class LangevinDynamicsComplete(Scene):
    def construct(self):
        # ========== PART 1: Introduction ==========
        # Title
        title = Text("Langevin Dynamics for Diffusion Models", font_size=42)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Introduction
        intro = Text("MCMC Method for Sampling", font_size=32)
        self.play(FadeIn(intro))
        self.wait(2)
        self.play(FadeOut(intro))
        
        # ========== PART 2: Equation 1 - Exponential Family ==========
        eq1_label = Text("Equation (1): Exponential Family", font_size=28)
        eq1_label.move_to(UP * 2.5)
        
        eq1 = MathTex(
            r"p(x) \propto e^{-f(x)}",
            font_size=80
        ).next_to(eq1_label, DOWN * 2)
        
        eq1_arrow = MathTex(r"\Rightarrow", font_size=80).next_to(eq1, DOWN)
        eq1_log = MathTex(
            r"\log p(x) = -f(x) + \text{const}",
            font_size=80
        ).next_to(eq1_arrow, RIGHT)
        
        eq1_group = VGroup(eq1, eq1_arrow, eq1_log).move_to(UP * 1)
        
        self.play(Write(eq1_label))
        self.play(Write(eq1))
        self.wait(2)
        self.play(Write(eq1_arrow), Write(eq1_log))
        self.wait(3)
        
        # Fade out equation 1
        self.play(FadeOut(eq1_label), FadeOut(eq1_group))
        
        # ========== PART 3: Key Assumption ==========
        assumption = Text("Key Assumption: We can only compute ∇f(x)", font_size=28)
        assumption.move_to(UP * 2.5)
        self.play(Write(assumption))
        self.wait(3)
        
        # ========== PART 4: Equation 2 - Gradient Descent ==========
        eq2_label = Text("Equation (2): Gradient Descent", font_size=28, color=YELLOW)
        eq2_label.move_to(UP * 1.5)
       
        eq2 = MathTex(
            r"x_t = x_{t-1} - \eta_t \nabla f(x_{t-1})",
            font_size=60
        ).next_to(eq2_label, DOWN)
        
        self.play(FadeOut(assumption))
        self.play(Write(eq2_label))
        self.play(Write(eq2))
        self.wait(2)
        
        # Show what GD does
        gd_text = Text("Finds modes (peaks) of the distribution", font_size=26, color=GREEN)
        gd_text.next_to(eq2, DOWN, buff=0.3)
        self.play(FadeIn(gd_text))
        self.wait(3)
        
        good_images = Text("Modes = 'Good' images in training data", font_size=24, color=GREEN)
        good_images.next_to(gd_text, DOWN, buff=0.2)
        self.play(FadeIn(good_images))
        self.wait(2)
        self.play(FadeOut(gd_text), FadeOut(good_images))
        
        # ========== PART 5: Equation 3 - Langevin Dynamics ==========
        eq3_label = Text("Equation (3): Langevin Dynamics", font_size=28, color=BLUE)
        eq3_label.move_to(DOWN * 0.5)
        
        eq3 = MathTex(
            r"x_{t+1} = x_t - \frac{\epsilon}{2} \nabla f(x_t) + \sqrt{\epsilon} \mathcal{N}(0, I)",
            font_size=60
        ).next_to(eq3_label, DOWN)
        
        self.play(Write(eq3_label))
        self.play(Write(eq3))
        self.wait(2)
        
        # Highlight the noise term
        noise_box = SurroundingRectangle(eq3[0][18:], color=RED, buff=0.1)
        noise_label = Text("Noise Term", font_size=24, color=RED)
        noise_label.next_to(noise_box, DOWN)
        
        self.play(Create(noise_box), Write(noise_label))
        self.wait(3)
        self.play(FadeOut(noise_box), FadeOut(noise_label))
        
        # Clear for visualization
        self.play(
            FadeOut(title),
            FadeOut(eq2_label),
            FadeOut(eq2),
            FadeOut(eq3_label),
            FadeOut(eq3)
        )
        self.wait(1)
        
        # ========== PART 6: Distribution Visualization Setup ==========
        # Create axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.2, 0.2],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": False}
        )
        
        # Create a bimodal distribution
        def prob_distribution(x):
            return 0.4 * np.exp(-((x + 1.5)**2) / 0.5) + 0.6 * np.exp(-((x - 1)**2) / 0.8)
        
        # Plot the distribution
        dist_graph = axes.plot(prob_distribution, color=BLUE, x_range=[-4, 4])
        
        viz_title = Text("Probability Distribution p(x)", font_size=36)
        viz_title.to_edge(UP)
        
        self.play(Create(axes), Write(viz_title))
        self.play(Create(dist_graph))
        self.wait(2)
        
        # ========== PART 7: Show Modes ==========
        mode1 = Dot(axes.c2p(-1.5, prob_distribution(-1.5)), color=YELLOW, radius=0.1)
        mode2 = Dot(axes.c2p(1, prob_distribution(1)), color=YELLOW, radius=0.1)
        
        mode_label = Text("Modes (Peaks)", font_size=28, color=YELLOW)
        mode_label.next_to(mode1, UP, buff=0.5)
        
        self.play(FadeIn(mode1), FadeIn(mode2))
        self.play(Write(mode_label))
        self.wait(3)
        
        # ========== PART 8: Gradient Descent Trajectory ==========
        gd_title = Text("Gradient Descent", font_size=32, color=GREEN)
        gd_title.to_edge(UP)
        
        self.play(Transform(viz_title, gd_title))
        
        # Starting point
        start_x = -3.5
        current_point = Dot(axes.c2p(start_x, 0), color=RED, radius=0.08)
        self.play(FadeIn(current_point))
        self.wait(1)
        
        # Simulate GD
        path_points = [axes.c2p(start_x, 0)]
        x = start_x
        eta = 0.3
        
        for _ in range(15):
            # Compute gradient numerically
            grad = (prob_distribution(x + 0.01) - prob_distribution(x - 0.01)) / 0.02
            x = x + eta * grad  # Going up the gradient (maximization)
            if x > 3.5:
                break
            path_points.append(axes.c2p(x, 0))
        
        path = VMobject(color=GREEN)
        path.set_points_as_corners(path_points)
        
        self.play(
            MoveAlongPath(current_point, path, rate_func=linear),
            Create(path),
            run_time=3
        )
        self.wait(1)
        
        converged = Text("Converges to nearest mode", font_size=24, color=GREEN)
        converged.next_to(current_point, DOWN)
        self.play(Write(converged))
        self.wait(3)
        
        # ========== PART 9: Clear for Langevin ==========
        self.play(
            FadeOut(current_point),
            FadeOut(path),
            FadeOut(converged),
            FadeOut(mode_label),
            FadeOut(mode1),
            FadeOut(mode2)
        )
        
        # ========== PART 10: Langevin Dynamics Simulation ==========
        langevin_title = Text("Langevin Dynamics", font_size=32, color=BLUE)
        langevin_title.to_edge(UP)
        self.play(Transform(viz_title, langevin_title))
        
        # Multiple particles with Langevin
        particles = VGroup(*[
            Dot(axes.c2p(-3.5 + i * 0.3, 0), color=BLUE, radius=0.06)
            for i in range(10)
        ])
        
        self.play(FadeIn(particles))
        self.wait(1)
        
        # Simulate Langevin dynamics
        epsilon = 0.2
        num_steps = 40
        
        for step in range(num_steps):
            new_positions = []
            for particle in particles:
                x_pos = axes.p2c(particle.get_center())[0]
                
                # Gradient
                grad = (prob_distribution(x_pos + 0.01) - prob_distribution(x_pos - 0.01)) / 0.02
                
                # Langevin update with noise
                noise = np.random.normal(0, 1)
                x_new = x_pos + (epsilon / 2) * grad + np.sqrt(epsilon) * noise * 0.3
                
                # Keep in bounds
                x_new = np.clip(x_new, -3.8, 3.8)
                new_positions.append(axes.c2p(x_new, 0))
            
            animations = [
                particle.animate.move_to(new_pos)
                for particle, new_pos in zip(particles, new_positions)
            ]
            
            self.play(*animations, run_time=0.1)
        
        self.wait(2)
        
        # ========== PART 11: Key Insight ==========
        explore_text = Text("Explores entire distribution!", font_size=28, color=BLUE)
        explore_text.to_edge(DOWN)
        self.play(Write(explore_text))
        self.wait(3)
        
        # Clear distribution visualization
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
        
        # ========== PART 12: Problem with Large Epsilon ==========
        problem_title = Text("Problem with Large ε", font_size=36, color=RED)
        problem_title.to_edge(UP)
        self.play(Write(problem_title))
        
        problem_text = Text(
            "High ε → Faster sampling but may diverge!",
            font_size=30,
            color=ORANGE
        )
        problem_text.move_to(UP * 1.5)
        self.play(Write(problem_text))
        self.wait(3)
        
        # Show equation with large epsilon highlighted
        eq3_again = MathTex(
            r"x_{t+1} = x_t - \frac{\epsilon}{2} \nabla f(x_t) + \sqrt{\epsilon} \mathcal{N}(0, I)",
            font_size=60
        ).move_to(UP * 0.3)
        
        self.play(Write(eq3_again))
        
        epsilon_box = SurroundingRectangle(eq3_again[0][8:9], color=RED)
        self.play(Create(epsilon_box))
        self.wait(3)
        
        # ========== PART 13: Solution ==========
        solution = Text("Solution: Metropolis-Hastings", font_size=32, color=GREEN)
        solution.move_to(DOWN * 1)
        
        guarantee = Text(
            "Provides convergence guarantees",
            font_size=26,
            color=GREEN
        )
        guarantee.next_to(solution, DOWN)
        
        self.play(Write(solution))
        self.play(Write(guarantee))
        self.wait(3)
        
        # Fade all
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(5)