import math
import os
import random
import time
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set


N = 8
ROOK_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "rock.png")


def initial_state_unassigned(n: int = N) -> List[int]:
    return [-1] * n


def is_goal(state: List[int]) -> bool:
    return -1 not in state and len(set(state)) == len(state)


def used_columns(state: List[int]) -> Set[int]:
    return {c for c in state if c != -1}


def next_unassigned_row(state: List[int]) -> Optional[int]:
    try:
        return state.index(-1)
    except ValueError:
        return None


def available_columns(state: List[int]) -> List[int]:
    n = len(state)
    used = used_columns(state)
    return [c for c in range(n) if c not in used]


def assign(state: List[int], row: int, col: int) -> List[int]:
    new_state = state.copy()
    new_state[row] = col
    return new_state


def conflicts_count(state: List[int]) -> int:
    cols = [c for c in state if c != -1]
    total = 0
    seen: Dict[int, int] = defaultdict(int)
    for c in cols:
        total += seen[c]
        seen[c] += 1
    return total


class Node:
    def __init__(self, state: List[int], g: int = 0):
        self.state = state
        self.g = g

    def __lt__(self, other: "Node") -> bool:
        return (self.g, self.state) < (other.g, other.state)


def expand(state: List[int]) -> List[List[int]]:
    row = next_unassigned_row(state)
    if row is None:
        return []
    return [assign(state, row, c) for c in available_columns(state)]


def bfs(n: int = N) -> Optional[List[int]]:
    start = initial_state_unassigned(n)
    q = deque([start])
    while q:
        s = q.popleft()
        if is_goal(s):
            return s
        for ns in expand(s):
            q.append(ns)
    return None


def dfs(n: int = N) -> Optional[List[int]]:
    start = initial_state_unassigned(n)
    stack = [start]
    while stack:
        s = stack.pop()
        if is_goal(s):
            return s
        stack.extend(expand(s))
    return None


def dls(limit: int, n: int = N) -> Optional[List[int]]:
    def rec(state: List[int], depth: int) -> Optional[List[int]]:
        if is_goal(state):
            return state
        if depth == limit:
            return None
        for ns in expand(state):
            sol = rec(ns, depth + 1)
            if sol is not None:
                return sol
        return None

    return rec(initial_state_unassigned(n), 0)


def ids(n: int = N) -> Optional[List[int]]:
    for limit in range(n + 1):
        sol = dls(limit, n)
        if sol is not None:
            return sol
    return None


def ucs(n: int = N) -> Optional[List[int]]:
    import heapq

    start = Node(initial_state_unassigned(n), g=0)
    heap: List[Tuple[int, int, Node]] = []
    counter = 0
    heapq.heappush(heap, (start.g, counter, start))
    while heap:
        _, _, node = heapq.heappop(heap)
        if is_goal(node.state):
            return node.state
        for ns in expand(node.state):
            counter += 1
            heapq.heappush(heap, (node.g + 1, counter, Node(ns, node.g + 1)))
    return None




def heuristic_remaining_conflicts(state: List[int]) -> int:
    return conflicts_count(state)


def greedy_best_first(n: int = N) -> Optional[List[int]]:
    import heapq

    start = initial_state_unassigned(n)
    heap: List[Tuple[int, int, List[int]]] = []
    counter = 0
    heapq.heappush(heap, (heuristic_remaining_conflicts(start), counter, start))
    while heap:
        _, _, s = heapq.heappop(heap)
        if is_goal(s):
            return s
        for ns in expand(s):
            counter += 1
            h = heuristic_remaining_conflicts(ns)
            heapq.heappush(heap, (h, counter, ns))
    return None


def a_star(n: int = N) -> Optional[List[int]]:
    import heapq

    start = Node(initial_state_unassigned(n), g=0)
    heap: List[Tuple[int, int, Node]] = []
    counter = 0
    heapq.heappush(heap, (heuristic_remaining_conflicts(start.state), counter, start))
    while heap:
        _, _, node = heapq.heappop(heap)
        if is_goal(node.state):
            return node.state
        for ns in expand(node.state):
            g = node.g + 1
            f = g + heuristic_remaining_conflicts(ns)
            counter += 1
            heapq.heappush(heap, (f, counter, Node(ns, g)))
    return None


def backtracking(n: int = N) -> Optional[List[int]]:
    def rec(state: List[int]) -> Optional[List[int]]:
        if is_goal(state):
            return state
        row = next_unassigned_row(state)
        for c in available_columns(state):
            sol = rec(assign(state, row, c))
            if sol is not None:
                return sol
        return None

    return rec(initial_state_unassigned(n))




def csp_forward_checking(n: int = N) -> Optional[List[int]]:
    domains: Dict[int, Set[int]] = {r: set(range(n)) for r in range(n)}
    assignment: Dict[int, int] = {}

    def forward(r: int, c: int, doms: Dict[int, Set[int]]) -> Optional[Dict[int, Set[int]]]:
        newdoms: Dict[int, Set[int]] = {k: v.copy() for k, v in doms.items()}
        for rr in range(n):
            if rr != r and c in newdoms[rr]:
                newdoms[rr].remove(c)
                if not newdoms[rr]:
                    return None
        return newdoms

    def select_unassigned_var(doms: Dict[int, Set[int]]) -> int:
        unassigned = [r for r in range(n) if r not in assignment]
        return min(unassigned, key=lambda r: len(doms[r]))

    def rec(doms: Dict[int, Set[int]]) -> Optional[Dict[int, int]]:
        if len(assignment) == n:
            return assignment.copy()
        r = select_unassigned_var(doms)
        for c in sorted(doms[r]):
            assignment[r] = c
            nd = forward(r, c, doms)
            if nd is not None:
                res = rec(nd)
                if res is not None:
                    return res
            assignment.pop(r)
        return None

    res = rec(domains)
    if res is None:
        return None
    return [res[r] for r in range(n)]


def ac3(arcs: List[Tuple[int, int]], domains: Dict[int, Set[int]]) -> bool:
    q = deque(arcs)
    def revise(x: int, y: int) -> bool:
        removed = False
        to_remove = set()
        for vx in domains[x]:
            if all(vy == vx for vy in domains[y]):
                to_remove.add(vx)
        if to_remove:
            domains[x] -= to_remove
            removed = True
        return removed

    while q:
        x, y = q.popleft()
        if revise(x, y):
            if not domains[x]:
                return False
            for z in domains.keys():
                if z != x and z != y:
                    q.append((z, x))
    return True


def csp_ac3_then_backtracking(n: int = N) -> Optional[List[int]]:
    domains: Dict[int, Set[int]] = {r: set(range(n)) for r in range(n)}
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if not ac3(arcs, domains):
        return None

    assignment: Dict[int, int] = {}

    def select_unassigned_var() -> int:
        unassigned = [r for r in range(n) if r not in assignment]
        return min(unassigned, key=lambda r: len(domains[r]))

    def rec() -> Optional[Dict[int, int]]:
        if len(assignment) == n:
            return assignment.copy()
        r = select_unassigned_var()
        for c in sorted(domains[r]):
            if all(assignment.get(rr) != c for rr in assignment):
                assignment[r] = c
                res = rec()
                if res is not None:
                    return res
                assignment.pop(r)
        return None

    res = rec()
    if res is None:
        return None
    return [res[r] for r in range(n)]




def random_state(n: int = N) -> List[int]:
    return [random.randrange(n) for _ in range(n)]


def total_conflicts(state: List[int]) -> int:
    return conflicts_count(state)


def neighbor_states(state: List[int]) -> List[List[int]]:
    n = len(state)
    neigh = []
    for r in range(n):
        for c in range(n):
            if c != state[r]:
                ns = state.copy()
                ns[r] = c
                neigh.append(ns)
    return neigh


def hill_climbing(max_steps: int = 1000, n: int = N) -> Optional[List[int]]:
    current = random_state(n)
    for _ in range(max_steps):
        if total_conflicts(current) == 0 and len(set(current)) == n:
            return current
        neigh = neighbor_states(current)
        best = min(neigh, key=total_conflicts)
        if total_conflicts(best) >= total_conflicts(current):
            current = [i for i in range(n)]
        else:
            current = best
    if total_conflicts(current) == 0 and len(set(current)) == n:
        return current
    return None


def simulated_annealing(n: int = N, max_steps: int = 2000, t0: float = 1.0, cooling: float = 0.995) -> Optional[List[int]]:
    current = random_state(n)
    t = t0
    for _ in range(max_steps):
        if total_conflicts(current) == 0 and len(set(current)) == n:
            return current
        r = random.randrange(n)
        c = random.randrange(n)
        next_state = current.copy()
        next_state[r] = c
        delta = total_conflicts(next_state) - total_conflicts(current)
        if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-6)):
            current = next_state
        t *= cooling
    if total_conflicts(current) == 0 and len(set(current)) == n:
        return current
    return None


def beam_search(k: int = 5, n: int = N, max_steps: int = 1000) -> Optional[List[int]]:
    beam = [random_state(n) for _ in range(k)]
    for _ in range(max_steps):
        beam.sort(key=total_conflicts)
        if total_conflicts(beam[0]) == 0 and len(set(beam[0])) == n:
            return beam[0]
        candidates: List[List[int]] = []
        for s in beam:
            candidates.extend(neighbor_states(s))
        candidates.sort(key=total_conflicts)
        beam = candidates[:k]
    beam.sort(key=total_conflicts)
    if total_conflicts(beam[0]) == 0 and len(set(beam[0])) == n:
        return beam[0]
    return None




def genetic_algorithm(n: int = N, pop_size: int = 50, generations: int = 200, mutation_rate: float = 0.1) -> Optional[List[int]]:
    def fitness(s: List[int]) -> int:
        conflicts = total_conflicts(s)
        max_pairs = n * (n - 1) // 2
        return max_pairs - conflicts

    def crossover(a: List[int], b: List[int]) -> List[int]:
        cut = random.randrange(1, n - 1)
        return a[:cut] + b[cut:]

    def mutate(s: List[int]) -> None:
        if random.random() < mutation_rate:
            r = random.randrange(n)
            s[r] = random.randrange(n)

    population = [random_state(n) for _ in range(pop_size)]
    for _ in range(generations):
        population.sort(key=lambda s: -fitness(s))
        best = population[0]
        if total_conflicts(best) == 0 and len(set(best)) == n:
            return best
        next_pop: List[List[int]] = population[: pop_size // 5]
        while len(next_pop) < pop_size:
            parents = random.sample(population[: pop_size // 2], 2)
            child = crossover(parents[0], parents[1])
            mutate(child)
            next_pop.append(child)
        population = next_pop
    population.sort(key=total_conflicts)
    if total_conflicts(population[0]) == 0 and len(set(population[0])) == n:
        return population[0]
    return None




def and_or_search(n: int = N) -> Optional[List[int]]:
    domains: Dict[int, Set[int]] = {r: set(range(n)) for r in range(n)}

    def and_step(assignment: Dict[int, int], doms: Dict[int, Set[int]]) -> Optional[Dict[int, int]]:
        if len(assignment) == n:
            return assignment
        r = min([rv for rv in range(n) if rv not in assignment], key=lambda rr: len(doms[rr]))
        for v in sorted(doms[r]):
            new_assignment = assignment.copy()
            new_assignment[r] = v
            new_doms: Dict[int, Set[int]] = {k: s.copy() for k, s in doms.items()}
            for rr in range(n):
                if rr != r and v in new_doms[rr]:
                    new_doms[rr].remove(v)
                    if not new_doms[rr]:
                        break
            else:
                res = and_step(new_assignment, new_doms)
                if res is not None:
                    return res
        return None

    res = and_step({}, domains)
    if res is None:
        return None
    return [res[r] for r in range(n)]




def run_algo(name: str, func, *args) -> Tuple[Optional[List[int]], float]:
    t0 = time.time()
    sol = func(*args)
    dt = (time.time() - t0) * 1000.0
    return sol, dt


def pretty_solution(sol: Optional[List[int]]) -> str:
    if sol is None:
        return "None"
    return "[" + ", ".join(str(x) for x in sol) + "]"


def main_console():
    algorithms = [
        ("BFS", bfs),
        ("DFS", dfs),
        ("DLS(depth=8)", lambda: dls(8)),
        ("IDS", ids),
        ("UCS", ucs),
        ("Greedy", greedy_best_first),
        ("A*", a_star),
        ("Backtracking", backtracking),
        ("Forward Checking", csp_forward_checking),
        ("AC-3 + BT", csp_ac3_then_backtracking),
        ("Hill Climbing", hill_climbing),
        ("Simulated Annealing", simulated_annealing),
        ("Beam(k=5)", lambda: beam_search(5)),
        ("Genetic", genetic_algorithm),
        ("AND-OR", and_or_search),
    ]

    print(f"Bai toan {N} con xe (N x N), ket qua la cot cua xe theo tung hang")
    for name, fn in algorithms:
        sol, ms = run_algo(name, fn)
        ok = sol is not None and len(set(sol)) == N
        print(f"{name:20s} | time: {ms:7.2f} ms | sol ok: {ok} | {pretty_solution(sol)}")


try:
    import tkinter as tk
except Exception:
    tk = None


class BoardUI:
    def __init__(self, root: "tk.Tk", n: int = N, selected_algorithm=None):
        self.root = root
        self.n = n
        self.cell = 60
        self.size = self.n * self.cell
        self.selected_algorithm = selected_algorithm
        
        root.configure(bg="#f8f9fa")
        header_frame = tk.Frame(root, bg="#2c3e50", height=60)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        if self.selected_algorithm:
            algo_name, _ = self.selected_algorithm
            tk.Label(header_frame, text=f"🏰 Đang chạy: {algo_name}", 
                    font=("Arial", 14, "bold"), fg="white", bg="#2c3e50").pack(expand=True)
        
        main_frame = tk.Frame(root, bg="#f8f9fa")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        board_frame = tk.Frame(main_frame, bg="#ffffff", relief="solid", bd=2)
        board_frame.pack(side="left", padx=(0, 10))
        
        self.canvas = tk.Canvas(board_frame, width=self.size + 25, height=self.size + 25, 
                               bg="#f8f9fa", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        control_panel = tk.Frame(main_frame, bg="#ffffff", relief="solid", bd=2, width=250)
        control_panel.pack(side="right", fill="y", padx=(10, 0))
        control_panel.pack_propagate(False)
        tk.Label(control_panel, text="⚙️ Điều khiển", 
                font=("Arial", 12, "bold"), fg="#2c3e50", bg="#ffffff").pack(pady=10)

        size_frame = tk.Frame(control_panel, bg="#ffffff")
        size_frame.pack(fill="x", padx=15, pady=5)
        tk.Label(size_frame, text="📏 Kích thước N:", 
                font=("Arial", 10, "bold"), fg="#34495e", bg="#ffffff").pack(anchor="w")
        
        self.n_var = tk.IntVar(value=self.n)
        self.spin = tk.Spinbox(size_frame, from_=4, to=16, textvariable=self.n_var, 
                              width=8, command=self.on_change_n, font=("Arial", 10),
                              bg="#ecf0f1", relief="solid", bd=1)
        self.spin.pack(anchor="w", pady=5)
        status_frame = tk.Frame(control_panel, bg="#ffffff")
        status_frame.pack(fill="x", padx=15, pady=5)
        tk.Label(status_frame, text="📊 Trạng thái:", 
                font=("Arial", 10, "bold"), fg="#34495e", bg="#ffffff").pack(anchor="w")
        
        self.status_var = tk.StringVar(value="🟢 Sẵn sàng")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               font=("Arial", 9), fg="#7f8c8d", bg="#ffffff", 
                               wraplength=200, justify="left")
        status_label.pack(anchor="w", pady=5)

        button_frame = tk.Frame(control_panel, bg="#ffffff")
        button_frame.pack(fill="x", padx=15, pady=10)
        reset_btn = tk.Button(button_frame, text="🔄 Reset", width=20, command=self.reset,
                             font=("Arial", 10, "bold"), bg="#95a5a6", fg="white",
                             relief="flat", bd=0, pady=8,
                             activebackground="#7f8c8d", activeforeground="white",
                             cursor="hand2")
        reset_btn.pack(pady=5)
        
        if self.selected_algorithm:
            algo_name, algo_func = self.selected_algorithm
            run_btn = tk.Button(button_frame, text=f"🚀 Chạy {algo_name}", width=20,
                               command=lambda: self.run_and_draw(algo_name, algo_func),
                               font=("Arial", 10, "bold"), bg="#3498db", fg="white",
                               relief="flat", bd=0, pady=8,
                               activebackground="#2980b9", activeforeground="white",
                               cursor="hand2")
            run_btn.pack(pady=5)

        back_btn = tk.Button(button_frame, text="🔙 Quay lại Menu", width=20,
                            command=self.back_to_menu, bg="#e67e22", fg="white",
                            font=("Arial", 10, "bold"), relief="flat", bd=0, pady=8,
                            activebackground="#d35400", activeforeground="white",
                            cursor="hand2")
        back_btn.pack(pady=5)

        self.draw_board(None)

        self._img_refs: List["tk.PhotoImage"] = []
        self.rook_image = self._load_rook_image()
        
        if self.selected_algorithm:
            algo_name, algo_func = self.selected_algorithm
            self.root.after(500, lambda: self.run_and_draw(algo_name, algo_func))

    def on_change_n(self):
        self.n = int(self.n_var.get())
        self.size = self.n * self.cell
        label_size = 25
        self.canvas.config(width=self.size + label_size, height=self.size + label_size)
        self._img_refs.clear()
        self.rook_image = self._load_rook_image()
        self.draw_board(None)

    def draw_board(self, sol: Optional[List[int]]):
        self.canvas.delete("all")
        
        label_size = 25
        self.canvas.config(width=self.size + label_size, height=self.size + label_size)
        for r in range(self.n):
            y = r * self.cell + self.cell / 2
            self.canvas.create_text(label_size/2, y + label_size, text=str(r+1), 
                                  fill="#2c3e50", font=("Arial", 12, "bold"))
        
        for c in range(self.n):
            x = c * self.cell + self.cell / 2
            self.canvas.create_text(x + label_size, label_size/2, text=chr(65+c), 
                                  fill="#2c3e50", font=("Arial", 12, "bold"))
        
        for r in range(self.n):
            for c in range(self.n):
                x0 = c * self.cell + label_size
                y0 = r * self.cell + label_size
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                
                if (r + c) % 2 == 0:
                    color = "#ffffff"
                    outline_color = "#000000"
                else:
                    color = "#8b4513"
                    outline_color = "#000000"
                
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=outline_color, width=1)
        
        if sol is None:
            return
            
        for r, col in enumerate(sol):
            if 0 <= col < self.n:
                cx = col * self.cell + self.cell / 2 + label_size
                cy = r * self.cell + self.cell / 2 + label_size
                
                if self.rook_image is not None:
                    img = self.rook_image
                    self._img_refs.append(img)
                    self.canvas.create_image(cx, cy, image=img)
                else:
                    radius = self.cell * 0.25
                    self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, 
                                          fill="#dc3545", outline="#c82333", width=3)
                    self.canvas.create_text(cx, cy, text="♜", fill="white", 
                                          font=("Arial", int(self.cell * 0.4), "bold"))

    def run_and_draw(self, name: str, func):
        self.status_var.set("🔄 Đang chạy thuật toán...")
        self.root.update()
        
        start = time.time()
        try:
            sol = func(self.n)
        except TypeError:
            sol = func()
        ms = (time.time() - start) * 1000.0
        
        ok = sol is not None and len(sol) == self.n and len(set(sol)) == self.n
        
        if ok:
            self.status_var.set(f"✅ {name}: {ms:.1f} ms | Kết quả hợp lệ")
        else:
            self.status_var.set(f"❌ {name}: {ms:.1f} ms | Không tìm thấy giải pháp")
            
        self.draw_board(sol)

    def reset(self):
        self.status_var.set("🟢 Đã reset bàn cờ")
        self.draw_board(None)


    def back_to_menu(self):
        self.root.destroy()
        show_algorithm_menu()

    def _load_rook_image(self) -> Optional["tk.PhotoImage"]:
        size = int(self.cell * 0.8)
        try:
            from PIL import Image, ImageTk 
            img = Image.open(ROOK_IMAGE_PATH).convert("RGBA")
            img = img.resize((size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            try:
                img = tk.PhotoImage(file=ROOK_IMAGE_PATH)
                w = max(1, img.width())
                factor = max(1, int(w / size))
                if factor > 1:
                    img = img.subsample(factor, factor)
                return img
            except Exception:
                return None


def show_algorithm_menu():
    if tk is None:
        main_console()
        return
    
    menu_root = tk.Tk()
    menu_root.title("🏰 Giải Bài Toán N Con Xe")
    menu_root.geometry("600x800")
    menu_root.resizable(False, False)
    menu_root.configure(bg="#f5f5f5")
    
    header_frame = tk.Frame(menu_root, bg="#2c3e50", height=80)
    header_frame.pack(fill="x")
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="🏰 GIẢI BÀI TOÁN N CON XE", 
             font=("Arial", 18, "bold"), fg="white", bg="#2c3e50").pack(expand=True)
    tk.Label(header_frame, text="Chọn thuật toán để giải quyết bài toán đặt xe trên bàn cờ", 
             font=("Arial", 10), fg="#ecf0f1", bg="#2c3e50").pack()
    radio_frame = tk.Frame(menu_root, bg="#f5f5f5")
    radio_frame.pack(fill="both", expand=True, padx=30, pady=20)
    
    tk.Label(radio_frame, text="📋 Danh sách thuật toán:", 
             font=("Arial", 12, "bold"), fg="#2c3e50", bg="#f5f5f5").pack(anchor="w", pady=(0, 10))
    
    algorithms = [
        ("BFS", bfs, "🔍 Tìm kiếm theo chiều rộng"),
        ("DFS", dfs, "🌳 Tìm kiếm theo chiều sâu"),
        ("DLS", lambda n: dls(n, n), "📏 Tìm kiếm giới hạn độ sâu"),
        ("IDS", ids, "🔄 Tìm kiếm tăng dần độ sâu"),
        ("UCS", ucs, "💰 Tìm kiếm chi phí đều"),
        ("Greedy", greedy_best_first, "🎯 Tìm kiếm tham lam"),
        ("A*", a_star, "⭐ Thuật toán A*"),
        ("Backtracking", backtracking, "🔙 Quay lui"),
        ("Forward Checking", csp_forward_checking, "✅ Kiểm tra tiến"),
        ("AC-3 + BT", csp_ac3_then_backtracking, "🔗 AC-3 + Quay lui"),
        ("Hill Climbing", hill_climbing, "⛰️ Leo đồi"),
        ("Simulated Annealing", simulated_annealing, "🔥 Ủ luyện"),
        ("Beam Search", lambda n: beam_search(5, n), "📡 Tìm kiếm chùm"),
        ("Genetic Algorithm", genetic_algorithm, "🧬 Thuật toán di truyền"),
        ("AND-OR Search", and_or_search, "🔀 Tìm kiếm AND-OR"),
    ]
    
    selected_algo = tk.StringVar()
    selected_algo.set("BFS")
    
    algorithms_frame = tk.Frame(radio_frame, bg="#ffffff", relief="solid", bd=1)
    algorithms_frame.pack(fill="both", expand=True)
    for i, (name, func, description) in enumerate(algorithms):
        row = i // 2
        col = i % 2
        
        frame = tk.Frame(algorithms_frame, bg="#ffffff", relief="solid", bd=1)
        frame.grid(row=row, column=col, sticky="ew", padx=5, pady=3)
        
        rb = tk.Radiobutton(frame, text=name, variable=selected_algo, value=name,
                           font=("Arial", 10, "bold"), bg="#ffffff", 
                           selectcolor="#3498db", fg="#2c3e50", 
                           activebackground="#3498db", activeforeground="white",
                           relief="flat", bd=0, indicatoron=0,
                           width=18, height=2)
        rb.pack(side="left", padx=8, pady=6)
        
        tk.Label(frame, text=description, font=("Arial", 8), 
                fg="#7f8c8d", bg="#ffffff").pack(side="left", padx=5, pady=6)
    
    algorithms_frame.grid_columnconfigure(0, weight=1)
    algorithms_frame.grid_columnconfigure(1, weight=1)
    
    def start_board():
        algo_name = selected_algo.get()
        algo_func = next(func for name, func, _ in algorithms if name == algo_name)
        menu_root.destroy()
        
        root = tk.Tk()
        root.title(f"🏰 8 con xe - {algo_name}")
        BoardUI(root, n=N, selected_algorithm=(algo_name, algo_func))
        root.mainloop()
    
    button_frame = tk.Frame(menu_root, bg="#f5f5f5", height=80)
    button_frame.pack(fill="x", pady=20)
    button_frame.pack_propagate(False)
    exit_btn = tk.Button(button_frame, text="❌ THOÁT", command=menu_root.destroy,
                        font=("Arial", 14, "bold"), bg="#e74c3c", fg="white",
                        padx=40, pady=15, relief="flat", bd=0,
                        activebackground="#c0392b", activeforeground="white",
                        cursor="hand2")
    exit_btn.pack(side="left", padx=20, expand=True)
    
    start_btn = tk.Button(button_frame, text="🚀 BẮT ĐẦU", command=start_board,
                         font=("Arial", 14, "bold"), bg="#27ae60", fg="white",
                         padx=40, pady=15, relief="flat", bd=0,
                         activebackground="#2ecc71", activeforeground="white",
                         cursor="hand2")
    start_btn.pack(side="right", padx=20, expand=True)
    
    menu_root.mainloop()


def main_ui():
    show_algorithm_menu()
 

if __name__ == "__main__":
    main_ui()
