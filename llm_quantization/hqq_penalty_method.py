from manim import *
import numpy as np

class OptimizationPenalty(Scene):
    def construct(self):
        # Title
        title = Text("Optimization using Penalty Method", font_size=48, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        
        # Use the same function throughout: f(x) = x^4 - 4x^2 + 3
        # This has minima at x = ±√2 ≈ ±1.414, with f(±√2) = -1
        
        # Scene 1: Non-Convex Functions
        axes, graph = self.non_convex_demo()
        self.wait(2)
        
        # Transition to Scene 2: Add constraint to the same function
        self.transition_to_constrained(axes, graph)
        self.wait(2)
        
        # Transition to Scene 3: Show penalty method on the same problem
        self.transition_to_penalty_method(axes)
        self.wait(2)

    def non_convex_demo(self):
        title = Text("1. Non-Convex Functions", font_size=36, color=RED).to_edge(UP)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 6, 2],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE}
        ).shift(DOWN * 0.5)
        
        # Our main function: f(x) = x^4 - 4x^2 + 3
        def main_func(x):
            return x**4 - 4*x**2 + 3
        
        # Plot the function
        graph = axes.plot(main_func, color=YELLOW, x_range=[-2.5, 2.5])
        
        self.play(Create(axes), Create(graph))
        
        # Mark critical points
        sqrt2 = np.sqrt(2)
        local_min1 = Dot(axes.coords_to_point(-sqrt2, main_func(-sqrt2)), color=RED, radius=0.08)
        local_min2 = Dot(axes.coords_to_point(sqrt2, main_func(sqrt2)), color=RED, radius=0.08)
        local_max = Dot(axes.coords_to_point(0, main_func(0)), color=GREEN, radius=0.08)
        
        self.play(Create(local_min1), Create(local_min2), Create(local_max))
        
        # Labels
        min_label = Text("Global Minima\nf(±√2) = -1", font_size=20, color=RED).to_edge(LEFT)
        max_label = Text("Local Maximum\nf(0) = 3", font_size=20, color=GREEN).next_to(min_label, DOWN, buff=0.5)
        
        self.play(Write(min_label), Write(max_label))
        
        # Explanation
        explanation = Text("Non-convex: Multiple local optima make optimization challenging", 
                         font_size=24, color=WHITE).to_edge(DOWN)
        self.play(Write(explanation))
        
        return axes, graph

    def transition_to_constrained(self, axes, graph):
        # Clear previous labels but keep the function
        self.play(FadeOut(*[mob for mob in self.mobjects if isinstance(mob, Text) and mob != graph]))
        
        # New title
        title = Text("2. Constrained Optimization", font_size=36, color=GREEN).to_edge(UP)
        self.play(Write(title))
        
        # Add constraint: x ≥ 0 (we can only consider the right half)
        constraint_line = axes.get_vertical_line(axes.coords_to_point(0, 0), color=RED, line_func=Line)
        constraint_region = Rectangle(
            width=axes.x_length/2, height=axes.y_length,
            fill_color=RED, fill_opacity=0.1, stroke_color=RED, stroke_width=2
        ).move_to(axes.coords_to_point(1.5, 2))
        
        self.play(Create(constraint_line), Create(constraint_region))
        
        # Highlight feasible region
        feasible_label = Text("Feasible Region: x ≥ 0", font_size=20, color=RED).to_edge(RIGHT).shift(UP)
        self.play(Write(feasible_label))
        
        # Show the constrained vs unconstrained optima
        sqrt2 = np.sqrt(2)
        unconstrained_opt = Dot(axes.coords_to_point(-sqrt2, ((-sqrt2)**4 - 4*(-sqrt2)**2 + 3)), 
                              color=GRAY, radius=0.08)
        constrained_opt = Dot(axes.coords_to_point(sqrt2, ((sqrt2)**4 - 4*(sqrt2)**2 + 3)), 
                           color=YELLOW, radius=0.08)
        boundary_opt = Dot(axes.coords_to_point(0, 3), color=ORANGE, radius=0.08)
        
        self.play(Create(unconstrained_opt), Create(constrained_opt), Create(boundary_opt))
        
        # Labels for optima
        opt_labels = VGroup(
            Text("Unconstrained optimum\n(infeasible)", font_size=18, color=GRAY),
            Text("Constrained optimum\nf(√2) = -1", font_size=18, color=YELLOW),
            Text("Boundary point\nf(0) = 3", font_size=18, color=ORANGE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT)
        
        self.play(Write(opt_labels))
        
        explanation = Text("Constraint x ≥ 0 eliminates the left minimum, changing our optimal solution", 
                         font_size=20, color=WHITE).to_edge(DOWN)
        self.play(Write(explanation))

    def transition_to_penalty_method(self, axes):
        # Clear everything except axes
        self.play(FadeOut(*[mob for mob in self.mobjects if mob != axes]))
        
        # New title
        title = Text("3. Penalty Method", font_size=36, color=PURPLE).to_edge(UP)
        self.play(Write(title))
        
        # Original function
        def main_func(x):
            return x**4 - 4*x**2 + 3
        
        # Penalty function: μ * max(0, -x)² for constraint x ≥ 0
        def penalty_func(x, mu):
            penalty = mu * np.maximum(0, -x)**2
            return main_func(x) + penalty
        
        # Show original function first
        original_graph = axes.plot(main_func, color=BLUE, x_range=[-2.5, 2.5])
        self.play(Create(original_graph))
        
        # Show constraint boundary
        constraint_line = axes.get_vertical_line(axes.coords_to_point(0, 0), color=RED, line_func=Line)
        constraint_text = Text("Constraint: x ≥ 0", font_size=20, color=RED).to_edge(RIGHT).shift(UP*2)
        self.play(Create(constraint_line), Write(constraint_text))
        
        original_label = Text("Original: f(x) = x⁴ - 4x² + 3", font_size=20, color=BLUE).to_edge(LEFT).shift(UP*2)
        self.play(Write(original_label))
        
        # Animate penalty method with increasing μ
        mu_values = [1, 10, 100]
        colors = [GREEN, ORANGE, PURPLE]
        
        penalty_graphs = []
        labels_group = VGroup()
        
        for i, (mu, color) in enumerate(zip(mu_values, colors)):
            penalty_graph = axes.plot(
                lambda x: penalty_func(x, mu), 
                color=color, 
                x_range=[-2.5, 2.5]
            )
            penalty_graphs.append(penalty_graph)
            
            # Create label for this penalty function
            mu_label = Text(f"μ = {mu}: f(x) + {mu}·max(0,-x)²", 
                          font_size=18, color=color).to_edge(LEFT).shift(UP*(1.5-i*0.4))
            labels_group.add(mu_label)
            
            self.play(Create(penalty_graph), Write(mu_label))
            
            # Show how the minimum shifts
            if mu == 1:
                min_dot = Dot(axes.coords_to_point(-0.8, penalty_func(-0.8, mu)), color=color, radius=0.06)
            elif mu == 10:
                min_dot = Dot(axes.coords_to_point(-0.3, penalty_func(-0.3, mu)), color=color, radius=0.06)
            else:  # mu == 100
                min_dot = Dot(axes.coords_to_point(0.05, penalty_func(0.05, mu)), color=color, radius=0.06)
            
            self.play(Create(min_dot))
            self.wait(0.8)
        
        # Final explanation
        explanation = VGroup(
            Text("As μ increases:", font_size=20, color=WHITE),
            Text("• Penalty for violating x ≥ 0 becomes severe", font_size=18, color=WHITE),
            Text("• Minimum shifts from x = -√2 toward x = 0", font_size=18, color=WHITE),
            Text("• Solution approaches the constrained optimum", font_size=18, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(DOWN).shift(UP*0.5)
        
        self.play(Write(explanation))
        
        # Show the progression with arrows
        arrow1 = Arrow(axes.coords_to_point(-1.4, 2), axes.coords_to_point(-0.8, 2), color=WHITE)
        arrow2 = Arrow(axes.coords_to_point(-0.8, 2), axes.coords_to_point(-0.3, 2), color=WHITE)
        arrow3 = Arrow(axes.coords_to_point(-0.3, 2), axes.coords_to_point(0.05, 2), color=WHITE)
        
        self.play(Create(arrow1), Create(arrow2), Create(arrow3))
        
        # Final summary
        summary = Text("Same function, three perspectives: Non-convex → Constrained → Penalty Method", 
                      font_size=24, color=GOLD).to_edge(DOWN)
        self.play(Transform(explanation, summary))


from manim import *

class LpNormNonConvex(Scene):
    def construct(self):
        title = Text("L^p Norm (p < 1) is Non-Convex", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 2, 0.5],
            axis_config={"color": BLUE}
        ).scale(1.2)
        labels = axes.get_axis_labels(x_label="x", y_label="||x||_p")
        self.play(Create(axes), Write(labels))

        p_values = [0.5, 1, 2]
        colors = [RED, GREEN, BLUE]
        graphs = []

        for i, p in enumerate(p_values):
            graph = axes.plot(lambda x: abs(x)**p, color=colors[i])
            graphs.append(graph)
            label = MathTex(f"p = {p}").next_to(graph, RIGHT)
            self.play(Create(graph), Write(label))

        explanation = Text("For p < 1, L^p norm forms a non-convex function", font_size=30)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))

        self.wait(3)

from manim import *

from manim import *
import numpy as np

class LpNormMultipleFunctions(ThreeDScene):
    def construct(self):
        # self.set_camera_orientation(phi=50 * DEGREES, theta=30 * DEGREES)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # 3D Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-5, 15, 5],
            x_length=6,
            y_length=6,
            z_length=6
        )
        self.play(Create(axes))

        # Add axis labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y") 
        z_label = axes.get_z_axis_label("z")
        self.play(Write(x_label), Write(y_label), Write(z_label))

        # Define function that returns 3D coordinates
        def surface_func(x, y, func):
            z_val = func(x, y)
            return np.array([x, y, z_val])

        def lp_norm_func(x, y, func, p):
            # Apply function to get z value
            z_val = func(x, y)
            # Create vector [x, y, z]
            vector = np.array([x, y, z_val])
            # Calculate Lp norm: (|x|^p + |y|^p + |z|^p)^(1/p)
            if p > 0:
                lp_norm_val = np.sum(np.abs(vector)**p)**(1/p)
            else:
                lp_norm_val = np.max(np.abs(vector))  # L∞ norm for p=0
            return np.array([x, y, lp_norm_val])

        # define lp norm function
        def lp_norm(x, y, p, func):
            f_xy = func(x, y)
            # f_xy = np.array([
                # x, y, func(x, y)
            # ])
            # f_xy = np.abs(f_xy)**p if p < 1 else (np.abs(f_xy)**p)**(1/p)
            lp_norm_val = np.array([x, y, f_xy])
            # print(lp_norm_val.shape)
            return lp_norm_val

        # Define different functions
        functions = {
            # "Linear: 3x + 4y": lambda x, y: 3*x + 4*y,
            "Quadratic: x² + y²": lambda x, y: x**2 + y**2,
            # "Absolute: |x| + |y|": lambda x, y: np.abs(x) + np.abs(y),
            # "Exponential: e^(x+y)": lambda x, y: np.exp(x + y),
            # "Sinusoidal: sin(x) + cos(y)": lambda x, y: np.sin(x) + np.cos(y)
        }

        # p_values = [2, 1, 0.5]
        colors = [BLUE, GREEN, RED]
        p_values = [0.5]

        # Iterate through functions and plot their Lp norms
        for idx, (func_name, func) in enumerate(functions.items()):
            for i, p in enumerate(p_values):
                graph = Surface(
                    lambda u, v: lp_norm(u, v, p, func),
                    u_range=[-1, 1],  # x range
                    v_range=[-1, 1],  # y range
                    checkerboard_colors=[RED_D, RED_E],
                    resolution=(20, 20)
                )

                # Lp norm transformed surface
                lp_graph = Surface(
                    lambda u, v: lp_norm_func(u, v, func, p),
                    u_range=[-1, 1],  # x range
                    v_range=[-1, 1],  # y range
                    checkerboard_colors=[BLUE_D, BLUE_E], 
                    resolution=(20, 20)
                )

                # Labels for both surfaces
                original_label = MathTex(f"Original: {func_name}", color=colors[0])
                original_label.to_corner(UR).shift(DOWN * 0.5)
                
                lp_label = MathTex(f"L_{p} norm: ||[x,y,f(x,y)]||_{p}", color=colors[1])
                lp_label.to_corner(UR).shift(DOWN * 1.2)
                
                # Animate original surface first
                self.play(Create(graph), Write(original_label))
                self.wait(2)
                
                # Then add the Lp norm surface
                self.play(Create(lp_graph), Write(lp_label))
                self.wait(2)

        # Explanation
        explanation = Text("Lp norm with p < 1 is non-convex!", font_size=30)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))

        # Rotate camera for better viewing
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(4)
        self.stop_ambient_camera_rotation()