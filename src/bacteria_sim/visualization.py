import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

class BacteriaVisualization:
    def __init__(self, model, update_interval=50, show_metrics=True, output_dir=None, save_interval=None):
        self.model = model
        self.update_interval = update_interval
        self.show_metrics = show_metrics
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_interval = save_interval
        
        # Configurar figura
        if show_metrics:
            self.fig = plt.figure(figsize=(12, 8))
            gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
            self.ax_cells = self.fig.add_subplot(gs[0, 0])
            self.ax_substrate = self.fig.add_subplot(gs[0, 1])
            self.ax_metrics = self.fig.add_subplot(gs[1, :])
        else:
            self.fig, (self.ax_cells, self.ax_substrate) = plt.subplots(1, 2, figsize=(10, 5))
            self.ax_metrics = None
        
        self.cmap_cells = ListedColormap(['white', 'blue', 'red'])
        self.bounds_cells = [0, 1, 2, 3]
        self.norm_cells = BoundaryNorm(self.bounds_cells, self.cmap_cells.N)
        
        self.im_cells = self.ax_cells.imshow(self.model.G, cmap=self.cmap_cells, norm=self.norm_cells, interpolation='nearest')
        self.im_substrate = self.ax_substrate.imshow(self.model.S, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        
        self.ax_cells.set_title("Estados celulares\nBlanco: vacío | Azul: división | Rojo: crecimiento")
        self.ax_substrate.set_title("Sustrato (blanco = disponible)")
        
        if self.ax_metrics:
            self.ax_metrics.set_xlabel("Paso")
            self.ax_metrics.set_ylabel("Cantidad")
            self.ax_metrics.grid(True, linestyle='--', alpha=0.7)
            self.line_cells, = self.ax_metrics.plot([], [], 'b-', label='Células')
            self.line_substrate, = self.ax_metrics.plot([], [], 'g-', label='Sustrato')
            self.ax_metrics.legend()
            self.time_steps = []
            self.history_cells = []
            self.history_substrate = []
        
        plt.tight_layout()
    
    def _update_metrics(self, step):
        if not self.ax_metrics:
            return
        total = self.model.get_cell_count()
        sub = self.model.get_substrate_amount()
        self.time_steps.append(step)
        self.history_cells.append(total)
        self.history_substrate.append(sub)
        self.line_cells.set_data(self.time_steps, self.history_cells)
        self.line_substrate.set_data(self.time_steps, self.history_substrate)
        self.ax_metrics.relim()
        self.ax_metrics.autoscale_view()
    
    def _update_display(self, step, total_steps):
        self.im_cells.set_data(self.model.G)
        self.im_substrate.set_data(self.model.S)
        self.ax_cells.set_title(f"Paso {step+1}/{total_steps}")
        self._update_metrics(step)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_frame(self, step):
        """Guarda el estado actual como imagen PNG en self.output_dir."""
        if not self.output_dir:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(self.model.G, cmap=self.cmap_cells, norm=self.norm_cells, interpolation='nearest')
        ax1.set_title(f"Células (paso {step})")
        ax2.imshow(self.model.S, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax2.set_title("Sustrato")
        filename = self.output_dir / f"bacteria_step_{step:04d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Guardado] {filename}")
    
    def animate_simulation(self, steps, save=False, filename="simulation.png"):
        try:
            for step in range(steps):
                self.model.step()
                self._update_display(step, steps)
                plt.pause(self.update_interval / 1000.0)
                
                # Guardar cada save_interval pasos
                if self.save_interval and self.save_interval > 0 and (step + 1) % self.save_interval == 0:
                    self.save_frame(step + 1)
                
                if not plt.fignum_exists(self.fig.number):
                    break
            
            if save:
                self.fig.savefig(filename)
                print(f"Último frame guardado como {filename}")
            
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nSimulación interrumpida.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            plt.close(self.fig)
    
    def close(self):
        plt.close(self.fig)