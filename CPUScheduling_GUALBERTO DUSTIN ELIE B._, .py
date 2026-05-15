"""
CPU Scheduling Algorithms Simulator
=====================================
A high-class Python GUI application simulating:
  1. First-Come, First-Served (FCFS)
  2. Shortest Job First (SJF) – Non-preemptive
  3. Shortest Remaining Time (SRT) – Preemptive
  4. Round Robin (RR)
  5. Priority Scheduling (Non-preemptive)
  6. Priority Scheduling with Round Robin

"""

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from collections import deque
import copy

# ─────────────────────────────────────────────
#  COLOR PALETTE  (dark industrial-tech theme)
# ─────────────────────────────────────────────
BG_DARK      = "#0d1117"
BG_CARD      = "#161b22"
BG_INPUT     = "#21262d"
ACCENT       = "#00d4aa"      # teal-mint
ACCENT2      = "#f78166"      # coral
ACCENT3      = "#79c0ff"      # blue
TEXT_MAIN    = "#e6edf3"
TEXT_SUB     = "#8b949e"
BORDER       = "#30363d"
HIGHLIGHT    = "#1f6feb"
SUCCESS      = "#3fb950"
WARNING      = "#d29922"

PROC_COLORS  = [
    "#00d4aa", "#f78166", "#79c0ff", "#d2a8ff",
    "#ffa657", "#7ee787", "#ff7b72", "#56d364",
    "#e3b341", "#a5d6ff", "#ffa198", "#b1f0c1",
]

# ─────────────────────────────────────────────
#  SCHEDULING ALGORITHMS
# ─────────────────────────────────────────────

class Process:
    """Represents a single process with all scheduling attributes."""
    def __init__(self, pid, arrival, burst, priority=0):
        self.pid      = pid
        self.arrival  = arrival
        self.burst    = burst
        self.priority = priority
        self.remaining = burst
        self.start    = -1
        self.finish   = -1
        self.waiting  = 0
        self.turnaround = 0


def compute_times(processes):
    """Calculate WT and TAT for each process after simulation."""
    for p in processes:
        p.turnaround = p.finish - p.arrival
        p.waiting    = p.turnaround - p.burst


def fcfs(processes):
    """
    First-Come, First-Served (FCFS) – Non-preemptive.
    Processes execute in order of arrival time.
    Returns timeline list of (pid, start, end).
    """
    procs = sorted(copy.deepcopy(processes), key=lambda x: (x.arrival, x.pid))
    timeline = []
    time = 0
    for p in procs:
        if time < p.arrival:
            time = p.arrival          # CPU idle gap
        p.start  = time
        p.finish = time + p.burst
        timeline.append((p.pid, p.start, p.finish))
        time = p.finish
    compute_times(procs)
    return procs, timeline


def sjf_non_preemptive(processes):
    """
    Shortest Job First (SJF) – Non-preemptive.
    At each decision point, pick the ready process with shortest burst.
    """
    procs = copy.deepcopy(processes)
    done  = []
    timeline = []
    time  = 0
    remaining = list(procs)

    while remaining:
        # Available processes that have arrived
        ready = [p for p in remaining if p.arrival <= time]
        if not ready:
            time = min(p.arrival for p in remaining)
            continue
        # Pick shortest burst (tie-break: arrival, then pid)
        chosen = min(ready, key=lambda x: (x.burst, x.arrival, x.pid))
        remaining.remove(chosen)
        chosen.start  = time
        chosen.finish = time + chosen.burst
        timeline.append((chosen.pid, chosen.start, chosen.finish))
        time = chosen.finish
        done.append(chosen)

    compute_times(done)
    return done, timeline


def srt_preemptive(processes):
    """
    Shortest Remaining Time (SRT) – Preemptive version of SJF.
    At every time unit, run the process with the shortest remaining burst.
    """
    procs   = copy.deepcopy(processes)
    n       = len(procs)
    time    = 0
    done    = 0
    timeline_raw = []

    for p in procs:
        p.remaining = p.burst

    max_time = sum(p.burst for p in procs) + max(p.arrival for p in procs) + 1

    while done < n and time < max_time:
        ready = [p for p in procs if p.arrival <= time and p.finish == -1 and p.remaining > 0]
        if not ready:
            time += 1
            continue
        chosen = min(ready, key=lambda x: (x.remaining, x.arrival, x.pid))
        if chosen.start == -1:
            chosen.start = time
        timeline_raw.append(chosen.pid)
        chosen.remaining -= 1
        time += 1
        if chosen.remaining == 0:
            chosen.finish = time
            done += 1

    # Compress timeline_raw into (pid, start, end) segments
    timeline = compress_timeline(timeline_raw)
    compute_times(procs)
    return procs, timeline


def round_robin(processes, quantum):
    """
    Round Robin (RR) – Preemptive with fixed time quantum.
    Each process gets at most 'quantum' units per turn.
    """
    procs   = copy.deepcopy(processes)
    for p in procs:
        p.remaining = p.burst

    queue    = deque()
    timeline_raw = []
    time     = 0
    arrived  = set()
    done     = 0
    n        = len(procs)
    procs_sorted = sorted(procs, key=lambda x: (x.arrival, x.pid))

    # Seed initial arrivals at time 0
    for p in procs_sorted:
        if p.arrival <= time:
            queue.append(p)
            arrived.add(p.pid)

    while done < n:
        if not queue:
            # Jump to next arrival
            future = [p for p in procs_sorted if p.pid not in arrived]
            if not future:
                break
            time = min(p.arrival for p in future)
            for p in procs_sorted:
                if p.arrival <= time and p.pid not in arrived:
                    queue.append(p)
                    arrived.add(p.pid)
            continue

        current = queue.popleft()
        if current.start == -1:
            current.start = time

        run = min(quantum, current.remaining)
        for _ in range(run):
            timeline_raw.append(current.pid)
            time += 1
            # Check for new arrivals at each tick
            for p in procs_sorted:
                if p.arrival <= time and p.pid not in arrived:
                    queue.append(p)
                    arrived.add(p.pid)

        current.remaining -= run
        if current.remaining == 0:
            current.finish = time
            done += 1
        else:
            queue.append(current)

    timeline = compress_timeline(timeline_raw)
    compute_times(procs)
    return procs, timeline


def priority_non_preemptive(processes, higher_is_better=False):
    """
    Priority Scheduling – Non-preemptive.
    Lower priority number = higher priority by default (can be inverted).
    At each decision point, pick ready process with best priority.
    """
    procs  = copy.deepcopy(processes)
    done   = []
    timeline = []
    time   = 0
    remaining = list(procs)

    while remaining:
        ready = [p for p in remaining if p.arrival <= time]
        if not ready:
            time = min(p.arrival for p in remaining)
            continue
        if higher_is_better:
            chosen = max(ready, key=lambda x: (x.priority, -x.arrival))
        else:
            chosen = min(ready, key=lambda x: (x.priority, x.arrival))
        remaining.remove(chosen)
        chosen.start  = time
        chosen.finish = time + chosen.burst
        timeline.append((chosen.pid, chosen.start, chosen.finish))
        time = chosen.finish
        done.append(chosen)

    compute_times(done)
    return done, timeline


def priority_round_robin(processes, quantum, higher_is_better=False):
    """
    Priority Scheduling with Round Robin.
    Processes grouped by priority; within same priority, RR is applied.
    """
    procs = copy.deepcopy(processes)
    for p in procs:
        p.remaining = p.burst

    timeline_raw = []
    time   = 0
    done   = 0
    n      = len(procs)
    arrived = set()
    procs_sorted = sorted(procs, key=lambda x: (x.arrival, x.pid))

    def best_priority(candidates):
        if higher_is_better:
            return max(c.priority for c in candidates)
        else:
            return min(c.priority for c in candidates)

    queue = deque()

    for p in procs_sorted:
        if p.arrival <= time:
            queue.append(p)
            arrived.add(p.pid)

    while done < n:
        if not queue:
            future = [p for p in procs_sorted if p.pid not in arrived]
            if not future:
                break
            time = min(p.arrival for p in future)
            for p in procs_sorted:
                if p.arrival <= time and p.pid not in arrived:
                    queue.append(p)
                    arrived.add(p.pid)
            continue

        # Find best priority among queued
        bp = best_priority(list(queue))
        # Reorder: move non-best-priority processes to back, best to front
        top   = [p for p in queue if p.priority == bp]
        rest  = [p for p in queue if p.priority != bp]
        queue = deque(top + rest)

        current = queue.popleft()
        if current.start == -1:
            current.start = time

        run = min(quantum, current.remaining)
        for _ in range(run):
            timeline_raw.append(current.pid)
            time += 1
            for p in procs_sorted:
                if p.arrival <= time and p.pid not in arrived:
                    queue.append(p)
                    arrived.add(p.pid)

        current.remaining -= run
        if current.remaining == 0:
            current.finish = time
            done += 1
        else:
            # Re-insert; new arrivals may have changed best priority
            queue.append(current)

    timeline = compress_timeline(timeline_raw)
    compute_times(procs)
    return procs, timeline


def compress_timeline(raw):
    """Convert a flat list of PIDs into compressed (pid, start, end) segments."""
    if not raw:
        return []
    segments = []
    start = 0
    current = raw[0]
    for i in range(1, len(raw)):
        if raw[i] != current:
            segments.append((current, start, i))
            start   = i
            current = raw[i]
    segments.append((current, start, len(raw)))
    return segments


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class CPUSchedulerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CPU Scheduling Simulator")
        self.geometry("1340x880")
        self.configure(bg=BG_DARK)
        self.minsize(1100, 720)

        self._setup_styles()
        self._build_ui()

    # ── Styles ──────────────────────────────
    def _setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Frame / Label
        self.style.configure("Dark.TFrame",       background=BG_DARK)
        self.style.configure("Card.TFrame",       background=BG_CARD)
        self.style.configure("Dark.TLabel",       background=BG_DARK,  foreground=TEXT_MAIN,
                             font=("Consolas", 10))
        self.style.configure("Card.TLabel",       background=BG_CARD,  foreground=TEXT_MAIN,
                             font=("Consolas", 10))
        self.style.configure("Title.TLabel",      background=BG_DARK,  foreground=ACCENT,
                             font=("Consolas", 20, "bold"))
        self.style.configure("Sub.TLabel",        background=BG_DARK,  foreground=TEXT_SUB,
                             font=("Consolas", 9))
        self.style.configure("SectionH.TLabel",   background=BG_CARD,  foreground=ACCENT3,
                             font=("Consolas", 11, "bold"))
        self.style.configure("Small.TLabel",      background=BG_CARD,  foreground=TEXT_SUB,
                             font=("Consolas", 8))

        # Notebook / Tabs
        self.style.configure("TNotebook",         background=BG_DARK,  borderwidth=0)
        self.style.configure("TNotebook.Tab",     background=BG_INPUT, foreground=TEXT_SUB,
                             font=("Consolas", 9, "bold"),
                             padding=[14, 6])
        self.style.map("TNotebook.Tab",
                       background=[("selected", BG_CARD)],
                       foreground=[("selected", ACCENT)])

        # Combobox
        self.style.configure("TCombobox",         fieldbackground=BG_INPUT,
                             background=BG_INPUT, foreground=TEXT_MAIN,
                             arrowcolor=ACCENT,   selectbackground=BG_INPUT,
                             font=("Consolas", 10))
        self.style.map("TCombobox",
                       fieldbackground=[("readonly", BG_INPUT)],
                       foreground=[("readonly", TEXT_MAIN)])

        # Scrollbar
        self.style.configure("Vertical.TScrollbar", background=BG_INPUT,
                             troughcolor=BG_DARK, arrowcolor=ACCENT)

        # Treeview (results table)
        self.style.configure("Result.Treeview",
                             background=BG_INPUT, foreground=TEXT_MAIN,
                             fieldbackground=BG_INPUT, rowheight=26,
                             font=("Consolas", 9))
        self.style.configure("Result.Treeview.Heading",
                             background=BG_CARD, foreground=ACCENT,
                             font=("Consolas", 9, "bold"))
        self.style.map("Result.Treeview",
                       background=[("selected", HIGHLIGHT)])

        # Separator
        self.style.configure("TSeparator", background=BORDER)

    # ── Master Layout ───────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK, height=64)
        hdr.pack(fill="x", padx=0, pady=0)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⚙ CPU SCHEDULING SIMULATOR",
                 bg=BG_DARK, fg=ACCENT,
                 font=("Consolas", 18, "bold")).pack(side="left", padx=24, pady=12)
        tk.Label(hdr, text="Operating Systems · Process Management",
                 bg=BG_DARK, fg=TEXT_SUB,
                 font=("Consolas", 9)).pack(side="left", padx=4, pady=20)

        # Separator line
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Main pane: left config + right output
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True)

        self._build_left(main)
        self._build_right(main)

    # ── LEFT PANEL ──────────────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG_CARD, width=380)
        left.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left.pack_propagate(False)

        # Scrollable canvas for left panel
        canvas = tk.Canvas(left, bg=BG_CARD, highlightthickness=0)
        sb     = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.left_inner = tk.Frame(canvas, bg=BG_CARD)
        win_id = canvas.create_window((0, 0), window=self.left_inner, anchor="nw")

        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=canvas.winfo_width())
        self.left_inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))

        self._bind_mousewheel(canvas)
        self._fill_left(self.left_inner)

    def _bind_mousewheel(self, widget):
        def _mw(e):
            widget.yview_scroll(int(-1*(e.delta/120)), "units")
        widget.bind_all("<MouseWheel>", _mw)

    def _fill_left(self, f):
        pad = dict(padx=16, pady=4)

        # ── Section: Algorithm ──
        self._section_label(f, "① ALGORITHM")
        self.algo_var = tk.StringVar(value="FCFS")
        algos = ["FCFS", "SJF (Non-Preemptive)", "SRT (Preemptive)",
                 "Round Robin", "Priority (Non-Preemptive)", "Priority + Round Robin"]
        cb = ttk.Combobox(f, textvariable=self.algo_var, values=algos,
                          state="readonly", width=34)
        cb.pack(**pad, fill="x")
        cb.bind("<<ComboboxSelected>>", self._on_algo_change)

        # Priority direction
        self.prio_frame = tk.Frame(f, bg=BG_CARD)
        self.prio_frame.pack(**pad, fill="x")
        tk.Label(self.prio_frame, text="Priority convention:",
                 bg=BG_CARD, fg=TEXT_SUB, font=("Consolas", 8)).pack(side="left")
        self.higher_is_better = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.prio_frame, text="Higher value = higher priority",
                        variable=self.higher_is_better,
                        style="TCheckbutton").pack(side="left", padx=6)
        self.prio_frame.pack_forget()

        # Quantum
        self.quantum_frame = tk.Frame(f, bg=BG_CARD)
        self.quantum_frame.pack(**pad, fill="x")
        tk.Label(self.quantum_frame, text="Time Quantum:",
                 bg=BG_CARD, fg=TEXT_SUB, font=("Consolas", 9)).pack(side="left")
        self.quantum_var = tk.StringVar(value="2")
        self._entry(self.quantum_frame, self.quantum_var, width=6).pack(side="left", padx=8)
        self.quantum_frame.pack_forget()

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=16, pady=8)

        # ── Section: Processes ──
        self._section_label(f, "② PROCESSES")

        ctrl = tk.Frame(f, bg=BG_CARD)
        ctrl.pack(**pad, fill="x")
        tk.Label(ctrl, text="Number of processes:",
                 bg=BG_CARD, fg=TEXT_SUB, font=("Consolas", 9)).pack(side="left")
        self.nproc_var = tk.StringVar(value="4")
        self._entry(ctrl, self.nproc_var, width=5).pack(side="left", padx=8)
        self._btn(ctrl, "Generate", self._generate_process_fields,
                  color=ACCENT).pack(side="left", padx=4)

        # Process table header
        self.proc_header = tk.Frame(f, bg=BG_CARD)
        self.proc_header.pack(padx=16, pady=(6, 0), fill="x")

        # Container for process rows
        self.proc_rows_frame = tk.Frame(f, bg=BG_CARD)
        self.proc_rows_frame.pack(padx=16, pady=2, fill="x")
        self.proc_rows = []   # list of dicts {pid, arrival, burst, priority}

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=16, pady=8)

        # ── Run button ──
        run_f = tk.Frame(f, bg=BG_CARD)
        run_f.pack(padx=16, pady=6, fill="x")
        self._btn(run_f, "▶  RUN SIMULATION", self._run_simulation,
                  color=ACCENT, font_size=11, height=2).pack(fill="x")

        self._btn(f, "⟳  Reset All", self._reset_all,
                  color=ACCENT2).pack(padx=16, pady=(0, 8), fill="x")

        # Initialise with default 4 processes
        self._generate_process_fields()

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg=BG_CARD, fg=ACCENT3,
                 font=("Consolas", 10, "bold")).pack(padx=16, pady=(12, 2), anchor="w")

    def _entry(self, parent, var, width=8):
        e = tk.Entry(parent, textvariable=var, width=width,
                     bg=BG_INPUT, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                     relief="flat", font=("Consolas", 10),
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT)
        return e

    def _btn(self, parent, text, cmd, color=ACCENT, font_size=9, height=1):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=BG_INPUT, fg=color, activebackground=BG_CARD,
                      activeforeground=color, relief="flat",
                      font=("Consolas", font_size, "bold"),
                      cursor="hand2", height=height,
                      highlightthickness=1, highlightbackground=color)
        b.bind("<Enter>", lambda e: b.config(bg=BG_CARD))
        b.bind("<Leave>", lambda e: b.config(bg=BG_INPUT))
        return b

    def _on_algo_change(self, *_):
        algo = self.algo_var.get()
        uses_quantum   = algo in ("Round Robin", "Priority + Round Robin")
        uses_priority  = algo in ("Priority (Non-Preemptive)", "Priority + Round Robin")

        if uses_quantum:
            self.quantum_frame.pack(padx=16, pady=4, fill="x",
                                    before=self.proc_header.master.winfo_children()[
                                        list(self.left_inner.winfo_children()).index(self.proc_rows_frame) - 2
                                    ] if False else None)
            self.quantum_frame.pack(padx=16, pady=4, fill="x")
        else:
            self.quantum_frame.pack_forget()

        if uses_priority:
            self.prio_frame.pack(padx=16, pady=4, fill="x")
        else:
            self.prio_frame.pack_forget()

        # Refresh rows to show/hide priority column
        self._refresh_proc_header(uses_priority)
        self._rebuild_proc_rows(uses_priority)

    def _refresh_proc_header(self, show_priority):
        for w in self.proc_header.winfo_children():
            w.destroy()
        cols = [("PID", 6), ("Arrival", 7), ("Burst", 7)]
        if show_priority:
            cols.append(("Priority", 7))
        for name, w in cols:
            tk.Label(self.proc_header, text=name, bg=BG_CARD, fg=ACCENT,
                     font=("Consolas", 8, "bold"), width=w).pack(side="left", padx=2)

    def _rebuild_proc_rows(self, show_priority):
        """Re-render rows keeping existing values."""
        existing = []
        for row in self.proc_rows:
            existing.append({
                "pid":      row["pid"].cget("text"),
                "arrival":  row["arrival_var"].get(),
                "burst":    row["burst_var"].get(),
                "priority": row["priority_var"].get() if "priority_var" in row else "0",
            })
        for w in self.proc_rows_frame.winfo_children():
            w.destroy()
        self.proc_rows.clear()
        for d in existing:
            self._add_proc_row(d["pid"], d["arrival"], d["burst"],
                               d["priority"], show_priority)

    def _generate_process_fields(self):
        try:
            n = int(self.nproc_var.get())
            if n < 3:
                messagebox.showwarning("Input Error", "Minimum 3 processes required.")
                return
            if n > 12:
                messagebox.showwarning("Input Error", "Maximum 12 processes supported.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Enter a valid integer for number of processes.")
            return

        algo = self.algo_var.get()
        uses_priority = algo in ("Priority (Non-Preemptive)", "Priority + Round Robin")
        self._refresh_proc_header(uses_priority)

        for w in self.proc_rows_frame.winfo_children():
            w.destroy()
        self.proc_rows.clear()

        # Default sample values
        defaults = [
            ("P1", "0", "6",  "2"),
            ("P2", "1", "8",  "1"),
            ("P3", "2", "7",  "3"),
            ("P4", "3", "3",  "4"),
            ("P5", "4", "4",  "2"),
            ("P6", "5", "5",  "5"),
            ("P7", "0", "2",  "1"),
            ("P8", "1", "9",  "3"),
            ("P9", "3", "4",  "2"),
            ("P10","2", "6",  "4"),
            ("P11","4", "3",  "1"),
            ("P12","5", "8",  "3"),
        ]
        for i in range(n):
            d = defaults[i] if i < len(defaults) else (f"P{i+1}", str(i), "5", "1")
            self._add_proc_row(d[0], d[1], d[2], d[3], uses_priority)

    def _add_proc_row(self, pid, arrival, burst, priority, show_priority):
        row_frame = tk.Frame(self.proc_rows_frame, bg=BG_CARD)
        row_frame.pack(fill="x", pady=2)

        pid_lbl = tk.Label(row_frame, text=pid, bg=BG_INPUT, fg=ACCENT,
                           font=("Consolas", 9, "bold"), width=6, relief="flat")
        pid_lbl.pack(side="left", padx=2)

        arr_var = tk.StringVar(value=arrival)
        bst_var = tk.StringVar(value=burst)
        pri_var = tk.StringVar(value=priority)

        self._entry(row_frame, arr_var, width=7).pack(side="left", padx=2)
        self._entry(row_frame, bst_var, width=7).pack(side="left", padx=2)

        row_data = {"pid": pid_lbl, "arrival_var": arr_var,
                    "burst_var": bst_var, "priority_var": pri_var,
                    "frame": row_frame}

        if show_priority:
            self._entry(row_frame, pri_var, width=7).pack(side="left", padx=2)

        self.proc_rows.append(row_data)

    # ── RIGHT PANEL ─────────────────────────
    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Gantt Chart
        self.gantt_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.gantt_tab, text="  📊 Gantt Chart  ")

        # Tab 2: Results Table
        self.table_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.table_tab, text="  📋 Results Table  ")

        # Tab 3: Statistics
        self.stats_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(self.stats_tab, text="  📈 Statistics  ")

        self._init_gantt_tab()
        self._init_table_tab()
        self._init_stats_tab()

    def _init_gantt_tab(self):
        placeholder = tk.Label(self.gantt_tab,
                               text="Run a simulation to view the Gantt Chart.",
                               bg=BG_DARK, fg=TEXT_SUB,
                               font=("Consolas", 11))
        placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self._gantt_placeholder = placeholder

    def _init_table_tab(self):
        # Summary cards row
        self.summary_frame = tk.Frame(self.table_tab, bg=BG_DARK)
        self.summary_frame.pack(fill="x", padx=10, pady=8)

        # Treeview
        cols = ("Process", "Arrival", "Burst", "Priority", "Start", "Finish",
                "Waiting Time", "Turnaround Time")
        frame = tk.Frame(self.table_tab, bg=BG_DARK)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")
        self.result_tree = ttk.Treeview(frame, columns=cols, show="headings",
                                         style="Result.Treeview",
                                         yscrollcommand=vsb.set,
                                         xscrollcommand=hsb.set)
        vsb.config(command=self.result_tree.yview)
        hsb.config(command=self.result_tree.xview)

        for c in cols:
            self.result_tree.heading(c, text=c)
            self.result_tree.column(c, width=100, anchor="center", minwidth=70)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.result_tree.pack(fill="both", expand=True)

    def _init_stats_tab(self):
        placeholder = tk.Label(self.stats_tab,
                               text="Run a simulation to view statistics.",
                               bg=BG_DARK, fg=TEXT_SUB,
                               font=("Consolas", 11))
        placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self._stats_placeholder = placeholder

    # ── SIMULATION RUNNER ───────────────────
    def _run_simulation(self):
        # Parse processes
        algo = self.algo_var.get()
        uses_priority = algo in ("Priority (Non-Preemptive)", "Priority + Round Robin")

        try:
            processes = []
            for row in self.proc_rows:
                pid     = row["pid"].cget("text")
                arrival = int(row["arrival_var"].get())
                burst   = int(row["burst_var"].get())
                prio    = int(row["priority_var"].get()) if uses_priority else 0
                if arrival < 0 or burst <= 0:
                    raise ValueError(f"{pid}: Arrival ≥ 0 and Burst > 0 required.")
                processes.append(Process(pid, arrival, burst, prio))
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        if len(processes) < 3:
            messagebox.showwarning("Input Error", "Minimum 3 processes required.")
            return

        # Parse quantum
        quantum = 2
        if algo in ("Round Robin", "Priority + Round Robin"):
            try:
                quantum = int(self.quantum_var.get())
                if quantum <= 0:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Input Error", "Time Quantum must be a positive integer.")
                return

        hib = self.higher_is_better.get()

        # Run chosen algorithm
        try:
            if algo == "FCFS":
                result, timeline = fcfs(processes)
            elif algo == "SJF (Non-Preemptive)":
                result, timeline = sjf_non_preemptive(processes)
            elif algo == "SRT (Preemptive)":
                result, timeline = srt_preemptive(processes)
            elif algo == "Round Robin":
                result, timeline = round_robin(processes, quantum)
            elif algo == "Priority (Non-Preemptive)":
                result, timeline = priority_non_preemptive(processes, hib)
            elif algo == "Priority + Round Robin":
                result, timeline = priority_round_robin(processes, quantum, hib)
            else:
                return
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            return

        self._render_gantt(timeline, result, algo)
        self._render_table(result, uses_priority)
        self._render_stats(result, algo)
        self.notebook.select(self.gantt_tab)

    def _render_gantt(self, timeline, processes, algo):
        # Clear old widgets
        for w in self.gantt_tab.winfo_children():
            w.destroy()

        pid_list = [p.pid for p in processes]
        color_map = {p.pid: PROC_COLORS[i % len(PROC_COLORS)]
                     for i, p in enumerate(processes)}

        if not timeline:
            tk.Label(self.gantt_tab, text="No timeline data.",
                     bg=BG_DARK, fg=TEXT_SUB, font=("Consolas", 11)).pack(expand=True)
            return

        max_t = max(seg[2] for seg in timeline)

        fig = plt.Figure(figsize=(11, 4.5), facecolor=BG_DARK)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_DARK)

        bar_height = 0.55
        y_center   = 0.5

        for seg in timeline:
            pid, start, end = seg
            color = color_map.get(pid, "#ffffff")
            ax.barh(y_center, end - start, left=start, height=bar_height,
                    color=color, edgecolor=BG_DARK, linewidth=1.5, alpha=0.92)
            mid = (start + end) / 2
            ax.text(mid, y_center, pid,
                    ha="center", va="center",
                    color=BG_DARK, fontweight="bold",
                    fontsize=9, fontfamily="Consolas")

        # Time ticks
        ticks = sorted(set(t for seg in timeline for t in [seg[1], seg[2]]))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks],
                           color=TEXT_SUB, fontsize=8, fontfamily="Consolas")
        ax.set_xlim(-0.3, max_t + 0.3)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(colors=TEXT_SUB, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

        ax.set_title(f"Gantt Chart — {algo}",
                     color=ACCENT, fontsize=12, fontfamily="Consolas",
                     fontweight="bold", pad=12)

        # Legend
        patches = [mpatches.Patch(color=color_map[p.pid], label=p.pid)
                   for p in processes]
        ax.legend(handles=patches, loc="upper right",
                  facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=TEXT_MAIN, fontsize=8,
                  prop={"family": "Consolas", "size": 8})

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        canvas = FigureCanvasTkAgg(fig, master=self.gantt_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        plt.close(fig)

    def _render_table(self, processes, show_priority):
        # Clear tree
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        # Clear summary cards
        for w in self.summary_frame.winfo_children():
            w.destroy()

        algo = self.algo_var.get()
        color_map = {p.pid: PROC_COLORS[i % len(PROC_COLORS)]
                     for i, p in enumerate(processes)}

        # Sort by PID for display
        sorted_procs = sorted(processes, key=lambda p: p.pid)

        tags_defined = set()
        for p in sorted_procs:
            tag = f"proc_{p.pid}"
            if tag not in tags_defined:
                self.result_tree.tag_configure(tag,
                    background=BG_INPUT,
                    foreground=color_map.get(p.pid, TEXT_MAIN))
                tags_defined.add(tag)
            vals = (p.pid, p.arrival, p.burst,
                    p.priority if show_priority else "—",
                    p.start, p.finish,
                    p.waiting, p.turnaround)
            self.result_tree.insert("", "end", values=vals, tags=(tag,))

        # Summary averages
        awt  = avg([p.waiting     for p in processes])
        atat = avg([p.turnaround  for p in processes])
        cpu_busy = sum(p.burst for p in processes)
        span = max(p.finish for p in processes) - min(p.arrival for p in processes)
        util = (cpu_busy / span * 100) if span > 0 else 100.0

        for label, val, color in [
            ("Avg Waiting Time",     f"{awt:.2f}",   ACCENT2),
            ("Avg Turnaround Time",  f"{atat:.2f}",  ACCENT3),
            ("CPU Utilization",      f"{util:.1f}%", SUCCESS),
            ("Total Processes",      str(len(processes)), ACCENT),
        ]:
            card = tk.Frame(self.summary_frame, bg=BG_CARD,
                            highlightthickness=1, highlightbackground=BORDER)
            card.pack(side="left", padx=6, pady=2, fill="both", expand=True)
            tk.Label(card, text=val, bg=BG_CARD, fg=color,
                     font=("Consolas", 16, "bold")).pack(pady=(8, 2))
            tk.Label(card, text=label, bg=BG_CARD, fg=TEXT_SUB,
                     font=("Consolas", 8)).pack(pady=(0, 8))

    def _render_stats(self, processes, algo):
        for w in self.stats_tab.winfo_children():
            w.destroy()

        fig = plt.Figure(figsize=(10, 5), facecolor=BG_DARK)
        color_map = {p.pid: PROC_COLORS[i % len(PROC_COLORS)]
                     for i, p in enumerate(processes)}
        pids    = [p.pid for p in processes]
        wts     = [p.waiting     for p in processes]
        tats    = [p.turnaround  for p in processes]
        colors  = [color_map[p.pid] for p in processes]

        x = range(len(pids))

        # WT bar chart
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(BG_DARK)
        bars = ax1.bar(x, wts, color=colors, edgecolor=BG_DARK, linewidth=1)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(pids, color=TEXT_SUB, fontfamily="Consolas", fontsize=9)
        ax1.set_title("Waiting Time per Process",
                      color=ACCENT2, fontfamily="Consolas", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Time Units", color=TEXT_SUB, fontfamily="Consolas", fontsize=9)
        ax1.tick_params(colors=TEXT_SUB)
        for spine in ax1.spines.values():
            spine.set_edgecolor(BORDER)
        ax1.axhline(avg(wts), color=ACCENT2, linestyle="--", linewidth=1, alpha=0.7,
                    label=f"Avg: {avg(wts):.2f}")
        ax1.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_MAIN,
                   prop={"family": "Consolas", "size": 8})
        for bar, val in zip(bars, wts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(val), ha="center", color=TEXT_MAIN,
                     fontsize=8, fontfamily="Consolas")

        # TAT bar chart
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(BG_DARK)
        bars2 = ax2.bar(x, tats, color=colors, edgecolor=BG_DARK, linewidth=1, alpha=0.85)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(pids, color=TEXT_SUB, fontfamily="Consolas", fontsize=9)
        ax2.set_title("Turnaround Time per Process",
                      color=ACCENT3, fontfamily="Consolas", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Time Units", color=TEXT_SUB, fontfamily="Consolas", fontsize=9)
        ax2.tick_params(colors=TEXT_SUB)
        for spine in ax2.spines.values():
            spine.set_edgecolor(BORDER)
        ax2.axhline(avg(tats), color=ACCENT3, linestyle="--", linewidth=1, alpha=0.7,
                    label=f"Avg: {avg(tats):.2f}")
        ax2.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_MAIN,
                   prop={"family": "Consolas", "size": 8})
        for bar, val in zip(bars2, tats):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(val), ha="center", color=TEXT_MAIN,
                     fontsize=8, fontfamily="Consolas")

        fig.suptitle(f"Performance Statistics — {algo}",
                     color=ACCENT, fontfamily="Consolas", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        canvas = FigureCanvasTkAgg(fig, master=self.stats_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        plt.close(fig)

    def _reset_all(self):
        self.algo_var.set("FCFS")
        self.nproc_var.set("4")
        self.quantum_var.set("2")
        self.higher_is_better.set(False)
        self.quantum_frame.pack_forget()
        self.prio_frame.pack_forget()
        self._generate_process_fields()

        # Clear tabs
        for w in self.gantt_tab.winfo_children():
            w.destroy()
        tk.Label(self.gantt_tab,
                 text="Run a simulation to view the Gantt Chart.",
                 bg=BG_DARK, fg=TEXT_SUB,
                 font=("Consolas", 11)).place(relx=0.5, rely=0.5, anchor="center")

        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for w in self.summary_frame.winfo_children():
            w.destroy()

        for w in self.stats_tab.winfo_children():
            w.destroy()
        tk.Label(self.stats_tab,
                 text="Run a simulation to view statistics.",
                 bg=BG_DARK, fg=TEXT_SUB,
                 font=("Consolas", 11)).place(relx=0.5, rely=0.5, anchor="center")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = CPUSchedulerApp()
    app.mainloop()