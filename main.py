import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
import subprocess
import threading
import webbrowser
import os
import time
from pathlib import Path

class FakeNewsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f2f5")
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f2f5')
        self.style.configure('TLabel', background='#f0f2f5', font=('Helvetica', 12))
        self.style.configure('TButton', font=('Helvetica', 12, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        
        # Server process
        self.server_process = None
        self.server_running = False
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.title_label = ttk.Label(
            self.header_frame, 
            text="Fake News Detection System", 
            style='Header.TLabel',
            font=('Helvetica', 24, 'bold')
        )
        self.title_label.pack(side=tk.LEFT, pady=10)
        
        # Create dataset frames
        self.datasets_frame = ttk.Frame(self.main_frame)
        self.datasets_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left dataset frame
        self.left_dataset_frame = ttk.Frame(self.datasets_frame)
        self.left_dataset_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.left_dataset_label = ttk.Label(
            self.left_dataset_frame, 
            text="Fake News Dataset", 
            style='Header.TLabel'
        )
        self.left_dataset_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.left_dataset_table = ttk.Treeview(self.left_dataset_frame)
        self.left_dataset_table.pack(fill=tk.BOTH, expand=True)
        
        # Right dataset frame
        self.right_dataset_frame = ttk.Frame(self.datasets_frame)
        self.right_dataset_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.right_dataset_label = ttk.Label(
            self.right_dataset_frame, 
            text="Real News Dataset", 
            style='Header.TLabel'
        )
        self.right_dataset_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.right_dataset_table = ttk.Treeview(self.right_dataset_frame)
        self.right_dataset_table.pack(fill=tk.BOTH, expand=True)
        
        # Console output
        self.console_frame = ttk.Frame(self.main_frame)
        self.console_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        self.console_label = ttk.Label(
            self.console_frame, 
            text="Console Output", 
            style='Header.TLabel'
        )
        self.console_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.console = scrolledtext.ScrolledText(
            self.console_frame, 
            height=10, 
            bg="#1e1e1e", 
            fg="#f0f0f0",
            font=("Consolas", 10)
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)
        
        # Buttons frame with gradient buttons
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(fill=tk.X, pady=20)
        
        # Create gradient canvas for Run Server button
        self.run_server_canvas = tk.Canvas(
            self.buttons_frame, 
            width=200, 
            height=40, 
            highlightthickness=0
        )
        self.run_server_canvas.pack(side=tk.LEFT, padx=(0, 20))
        
        # Create gradient for Run Server button
        self.run_server_gradient = self.create_gradient(
            self.run_server_canvas, 
            200, 40, 
            "#4e54c8", "#8f94fb"
        )
        
        # Create Run Server button text
        self.run_server_text = self.run_server_canvas.create_text(
            100, 20, 
            text="Run Server", 
            fill="white", 
            font=("Helvetica", 12, "bold")
        )
        
        # Bind click event to Run Server button
        self.run_server_canvas.tag_bind(self.run_server_gradient, "<Button-1>", self.toggle_server)
        self.run_server_canvas.tag_bind(self.run_server_text, "<Button-1>", self.toggle_server)
        
        # Create gradient canvas for Open Web Interface button
        self.open_web_canvas = tk.Canvas(
            self.buttons_frame, 
            width=200, 
            height=40, 
            highlightthickness=0
        )
        self.open_web_canvas.pack(side=tk.LEFT)
        
        # Create gradient for Open Web Interface button
        self.open_web_gradient = self.create_gradient(
            self.open_web_canvas, 
            200, 40, 
            "#11998e", "#38ef7d"
        )
        
        # Create Open Web Interface button text
        self.open_web_text = self.open_web_canvas.create_text(
            100, 20, 
            text="Open Web Interface", 
            fill="white", 
            font=("Helvetica", 12, "bold")
        )
        
        # Bind click event to Open Web Interface button
        self.open_web_canvas.tag_bind(self.open_web_gradient, "<Button-1>", self.open_web_interface)
        self.open_web_canvas.tag_bind(self.open_web_text, "<Button-1>", self.open_web_interface)
        
        # Status indicator
        self.status_frame = ttk.Frame(self.buttons_frame)
        self.status_frame.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(
            self.status_frame, 
            text="Server Status: ", 
            style='TLabel'
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.status_indicator = tk.Canvas(
            self.status_frame, 
            width=15, 
            height=15, 
            highlightthickness=0,
            bg="#f0f2f5"
        )
        self.status_indicator.pack(side=tk.LEFT)
        self.status_indicator.create_oval(2, 2, 13, 13, fill="red", outline="")
        
        # Load datasets
        self.load_datasets()
        
        # Add hover effects
        self.add_button_hover_effects()
        
        # Log startup
        self.log("Fake News Detection System started")
        self.log("Please load datasets and run the server to begin")

    def create_gradient(self, canvas, width, height, color1, color2):
        """Create a gradient rectangle on the canvas"""
        # Create gradient
        for i in range(height):
            # Calculate color for this line
            r1, g1, b1 = self.hex_to_rgb(color1)
            r2, g2, b2 = self.hex_to_rgb(color2)
            
            # Linear interpolation
            t = i / height
            r = r1 * (1 - t) + r2 * t
            g = g1 * (1 - t) + g2 * t
            b = b1 * (1 - t) + b2 * t
            
            color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
            canvas.create_line(0, i, width, i, fill=color)
        
        # Create rectangle with rounded corners
        return canvas.create_rectangle(
            0, 0, width, height, 
            fill="", 
            outline="", 
            width=0
        )

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def add_button_hover_effects(self):
        """Add hover effects to buttons"""
        # Run Server button hover effect
        self.run_server_canvas.tag_bind(
            self.run_server_gradient, 
            "<Enter>", 
            lambda e: self.button_hover(self.run_server_canvas, "#5d63d0", "#9ea3ff")
        )
        self.run_server_canvas.tag_bind(
            self.run_server_gradient, 
            "<Leave>", 
            lambda e: self.button_hover(self.run_server_canvas, "#4e54c8", "#8f94fb")
        )
        self.run_server_canvas.tag_bind(
            self.run_server_text, 
            "<Enter>", 
            lambda e: self.button_hover(self.run_server_canvas, "#5d63d0", "#9ea3ff")
        )
        self.run_server_canvas.tag_bind(
            self.run_server_text, 
            "<Leave>", 
            lambda e: self.button_hover(self.run_server_canvas, "#4e54c8", "#8f94fb")
        )
        
        # Open Web Interface button hover effect
        self.open_web_canvas.tag_bind(
            self.open_web_gradient, 
            "<Enter>", 
            lambda e: self.button_hover(self.open_web_canvas, "#0faa81", "#42ff8b")
        )
        self.open_web_canvas.tag_bind(
            self.open_web_gradient, 
            "<Leave>", 
            lambda e: self.button_hover(self.open_web_canvas, "#11998e", "#38ef7d")
        )
        self.open_web_canvas.tag_bind(
            self.open_web_text, 
            "<Enter>", 
            lambda e: self.button_hover(self.open_web_canvas, "#0faa81", "#42ff8b")
        )
        self.open_web_canvas.tag_bind(
            self.open_web_text, 
            "<Leave>", 
            lambda e: self.button_hover(self.open_web_canvas, "#11998e", "#38ef7d")
        )

    def button_hover(self, canvas, color1, color2):
        """Update button gradient on hover"""
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        # Clear canvas
        canvas.delete("all")
        
        # Recreate gradient
        for i in range(height):
            # Calculate color for this line
            r1, g1, b1 = self.hex_to_rgb(color1)
            r2, g2, b2 = self.hex_to_rgb(color2)
            
            # Linear interpolation
            t = i / height
            r = r1 * (1 - t) + r2 * t
            g = g1 * (1 - t) + g2 * t
            b = b1 * (1 - t) + b2 * t
            
            color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
            canvas.create_line(0, i, width, i, fill=color)
        
        # Recreate rectangle and text
        if canvas == self.run_server_canvas:
            self.run_server_gradient = canvas.create_rectangle(
                0, 0, width, height, 
                fill="", 
                outline="", 
                width=0
            )
            self.run_server_text = canvas.create_text(
                width // 2, height // 2, 
                text="Run Server" if not self.server_running else "Stop Server", 
                fill="white", 
                font=("Helvetica", 12, "bold")
            )
            
            # Rebind events
            canvas.tag_bind(self.run_server_gradient, "<Button-1>", self.toggle_server)
            canvas.tag_bind(self.run_server_text, "<Button-1>", self.toggle_server)
        else:
            self.open_web_gradient = canvas.create_rectangle(
                0, 0, width, height, 
                fill="", 
                outline="", 
                width=0
            )
            self.open_web_text = canvas.create_text(
                width // 2, height // 2, 
                text="Open Web Interface", 
                fill="white", 
                font=("Helvetica", 12, "bold")
            )
            
            # Rebind events
            canvas.tag_bind(self.open_web_gradient, "<Button-1>", self.open_web_interface)
            canvas.tag_bind(self.open_web_text, "<Button-1>", self.open_web_interface)

    def load_datasets(self):
        """Load and display datasets"""
        try:
            # Find dataset files
            dataset_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            dataset_files = list(dataset_dir.glob("*.csv"))
            
            if len(dataset_files) >= 2:
                # Load first dataset (real news)
                self.load_dataset_to_table(
                    str(dataset_files[0]), 
                    self.left_dataset_table, 
                    f"Real News Dataset: {dataset_files[0].name}"
                )
                
                # Load second dataset (fake news)
                self.load_dataset_to_table(
                    str(dataset_files[1]), 
                    self.right_dataset_table, 
                    f"Fake News Dataset: {dataset_files[1].name}"
                )
                
                self.log(f"Loaded datasets: {dataset_files[0].name} and {dataset_files[1].name}")
            elif len(dataset_files) == 1:
                # Load single dataset
                self.load_dataset_to_table(
                    str(dataset_files[0]), 
                    self.left_dataset_table, 
                    f"Dataset: {dataset_files[0].name}"
                )
                self.log(f"Loaded dataset: {dataset_files[0].name}")
                self.log("Warning: Only one dataset found. Expected two datasets.")
            else:
                self.log("No CSV datasets found in the current directory.")
                self.log("Please place your datasets in the same directory as this application.")
        except Exception as e:
            self.log(f"Error loading datasets: {str(e)}")

    def load_dataset_to_table(self, file_path, table, label_text):
        """Load a dataset into a table widget"""
        try:
            # Update label
            if table == self.left_dataset_table:
                self.left_dataset_label.config(text=label_text)
            else:
                self.right_dataset_label.config(text=label_text)
            
            # Load dataset
            df = pd.read_csv(file_path)
            
            # Clear existing table
            for item in table.get_children():
                table.delete(item)
            
            # Configure columns
            table['columns'] = list(df.columns)
            table['show'] = 'headings'
            
            for col in df.columns:
                table.heading(col, text=col)
                # Set column width based on content
                max_width = max(len(str(col)), df[col].astype(str).str.len().max())
                table.column(col, width=min(150, max(50, max_width * 7)))
            
            # Add data rows (limit to 10)
            for i, row in df.head(10).iterrows():
                values = [str(row[col])[:50] + '...' if len(str(row[col])) > 50 else str(row[col]) 
                          for col in df.columns]
                table.insert('', 'end', values=values)
                
        except Exception as e:
            self.log(f"Error loading dataset {file_path}: {str(e)}")

    def toggle_server(self, event=None):
        """Toggle server on/off"""
        if self.server_running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self):
        """Start the server in a separate thread"""
        if self.server_running:
            self.log("Server is already running")
            return
        
        self.log("Starting server...")
        
        # Update button text
        self.run_server_canvas.itemconfig(
            self.run_server_text, 
            text="Stop Server"
        )
        
        # Update status indicator
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
        
        # Set server running flag
        self.server_running = True
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=self._run_server)
        server_thread.daemon = True
        server_thread.start()

    def _run_server(self):
        """Run the server process"""
        try:
            server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classic_server.py")
            
            # Start server process
            self.server_process = subprocess.Popen(
                ["python", server_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read and log output
            for line in self.server_process.stdout:
                self.log(line.strip())
                
            # Process completed
            self.server_process.wait()
            
            # Update UI if process ends
            if self.server_running:
                self.root.after(0, self.stop_server)
                
        except Exception as e:
            self.log(f"Error running server: {str(e)}")
            self.root.after(0, self.stop_server)

    def stop_server(self):
        """Stop the server"""
        if not self.server_running:
            return
            
        self.log("Stopping server...")
        
        # Kill server process
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process = None
            except Exception as e:
                self.log(f"Error stopping server: {str(e)}")
        
        # Update button text
        self.run_server_canvas.itemconfig(
            self.run_server_text, 
            text="Run Server"
        )
        
        # Update status indicator
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 13, 13, fill="red", outline="")
        
        # Set server running flag
        self.server_running = False
        
        self.log("Server stopped")

    def open_web_interface(self, event=None):
        """Open the web interface in a browser"""
        try:
            self.log("Opening web interface...")
            
            # Check if server is running
            if not self.server_running:
                self.log("Warning: Server is not running. Starting server first...")
                self.start_server()
                # Give server time to start
                time.sleep(2)
            
          

# Register Chrome as the browser
            chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

            # Register and open the file in Chrome
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
            webbrowser.get('chrome').open(r"C:\Users\jkrut\OneDrive\Desktop\Fake_News\templates\index.html")
            
            self.log("Web interface opened in browser")
        except Exception as e:
            self.log(f"Error opening web interface: {str(e)}")

    def log(self, message):
        """Add a message to the console"""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsGUI(root)
    root.mainloop()