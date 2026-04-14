import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import sys
import queue
from pathlib import Path

# Fix path for local imports if run from project root
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

# Try importing configuration dataset lists for dropdown suggestions.
try:
    from config import IMAGE_DATASETS, LABEL_DATASETS
    KNOWN_IMAGE_DS = list(IMAGE_DATASETS.keys())
    KNOWN_LABEL_DS = list(LABEL_DATASETS.keys())
except ImportError:
    KNOWN_IMAGE_DS = []
    KNOWN_LABEL_DS = []

class TrainingUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rethinking Generalization - Training UI")
        self.geometry("850x650")
        
        self.process = None
        self.log_queue = queue.Queue()
        
        self.create_widgets()
        self.check_queue()

    def create_widgets(self):
        # --- Top Frame for Inputs ---
        input_frame = ttk.LabelFrame(self, text="Training Parameters", padding=(10, 10))
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Row 0: Model Type
        ttk.Label(input_frame, text="Model Type:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.model_var = tk.StringVar(value="MLP")
        self.model_combo = ttk.Combobox(input_frame, textvariable=self.model_var, values=["MLP", "ALEXNET"])
        self.model_combo.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Row 1: Image Dataset
        ttk.Label(input_frame, text="Image Dataset:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.image_var = tk.StringVar(value="cifar10_images_randomized")
        self.image_combo = ttk.Combobox(input_frame, textvariable=self.image_var, values=KNOWN_IMAGE_DS)
        self.image_combo.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Row 2: Label Dataset
        ttk.Label(input_frame, text="Label Dataset:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.label_var = tk.StringVar(value="cifar10")
        self.label_combo = ttk.Combobox(input_frame, textvariable=self.label_var, values=KNOWN_LABEL_DS)
        self.label_combo.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Row 3: Weight Decay & Dropout
        ttk.Label(input_frame, text="Weight Decay:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.wd_var = tk.StringVar(value="0.0")
        self.wd_entry = ttk.Entry(input_frame, textvariable=self.wd_var)
        self.wd_entry.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=5)
        
        ttk.Label(input_frame, text="Dropout Rate:").grid(row=3, column=2, sticky=tk.W, pady=5, padx=5)
        self.do_var = tk.StringVar(value="0.0")
        self.do_entry = ttk.Entry(input_frame, textvariable=self.do_var)
        self.do_entry.grid(row=3, column=3, sticky=tk.EW, pady=5, padx=5)
        
        # Row 4: Epochs & Batch Size
        ttk.Label(input_frame, text="Epochs:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        self.epochs_var = tk.StringVar(value="100")
        self.epochs_entry = ttk.Entry(input_frame, textvariable=self.epochs_var)
        self.epochs_entry.grid(row=4, column=1, sticky=tk.EW, pady=5, padx=5)
        
        ttk.Label(input_frame, text="Batch Size:").grid(row=4, column=2, sticky=tk.W, pady=5, padx=5)
        self.batch_var = tk.StringVar(value="128")
        self.batch_entry = ttk.Entry(input_frame, textvariable=self.batch_var)
        self.batch_entry.grid(row=4, column=3, sticky=tk.EW, pady=5, padx=5)
        
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)

        # Build command display preview
        self.cmd_var = tk.StringVar()
        self.update_cmd_display()
        for var in [self.model_var, self.image_var, self.label_var, self.wd_var, self.do_var, self.epochs_var, self.batch_var]:
            var.trace_add("write", lambda *args: self.update_cmd_display())

        cmd_frame = ttk.Frame(self)
        cmd_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        ttk.Label(cmd_frame, text="Preview:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        ttk.Entry(cmd_frame, textvariable=self.cmd_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # --- Middle Frame for Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="■ Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear Output", command=self.clear_output)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)

        # --- Bottom Frame for Output Log ---
        output_frame = ttk.LabelFrame(self, text="Training Output Logger", padding=(5, 5))
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.text_area = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(output_frame, command=self.text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=scrollbar.set)
        
    def update_cmd_display(self):
        cmd = ["python", "train.py", 
               self.model_var.get().strip(), 
               self.image_var.get().strip(), 
               self.label_var.get().strip(), 
               self.wd_var.get().strip(), 
               self.do_var.get().strip(), 
               self.epochs_var.get().strip(), 
               self.batch_var.get().strip()]
        # Remove empty items simply for a cleaner preview
        cmd = [c for c in cmd if c]
        self.cmd_var.set(" ".join(cmd))

    def write_output(self, text):
        self.log_queue.put(text)
        
    def check_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.text_area.config(state=tk.NORMAL)
                self.text_area.insert(tk.END, msg)
                self.text_area.see(tk.END)
                self.text_area.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(50, self.check_queue) # Polling ms

    def read_process_output(self, process):
        # Read lines sequentially from stdout
        for line in iter(process.stdout.readline, ''):
            self.write_output(line)
            
        process.stdout.close()
        process.wait()
        
        if process.returncode is not None:
            if process.returncode == 0:
                self.write_output(f"\n[INFO] Process finished successfully (exit code {process.returncode}).\n")
            else:
                self.write_output(f"\n[ERROR] Process stopped with exit code {process.returncode}.\n")
                
        self.after(0, self.on_process_finish)

    def start_training(self):
        if self.process is not None and self.process.poll() is None:
            return
            
        script_dir = Path(__file__).resolve().parent
        train_script = script_dir / "train.py"
        
        # --- SMART INTERPRETER DETECTION ---
        # If current python is not in a venv, try to find the project's .venv
        python_exe = sys.executable
        potential_venv = script_dir.parents[2] / ".venv" / "bin" / "python"
        if ".venv" not in python_exe and potential_venv.exists():
            python_exe = str(potential_venv)

        # Include -u flag to prevent python from buffering stdout so logs appear correctly.
        cmd = [
            python_exe, "-u", str(train_script),
            self.model_var.get().strip(),
            self.image_var.get().strip(),
            self.label_var.get().strip(),
            self.wd_var.get().strip(),
            self.do_var.get().strip(),
            self.epochs_var.get().strip(),
            self.batch_var.get().strip()
        ]
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.write_output(f"--- Starting Thread ---\n> {' '.join(cmd)}\n\n")
        
        # Launch subprocess capturing pipe outputs
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(script_dir)
        )
        
        # Read output in background
        threading.Thread(target=self.read_process_output, args=(self.process,), daemon=True).start()

    def stop_training(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.write_output("\n[INFO] Terminating background process...\n")
            
    def on_process_finish(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.process = None

    def clear_output(self):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = TrainingUI()
    app.mainloop()
