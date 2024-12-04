"""
This code creates a Light-Up Puzzle game using the Python tkinter library for graphical interface. 
The puzzle is grid-based, and the goal is to place light bulbs in white cells to illuminate 
the entire grid, while respecting constraints on black cells with numbers. 

Note: some of the codebase are generated using ChatGPT-4o

Instructions:

Add a numbered black cell:

1. Press a number key between 0 and 4.
Click on the cell where you want to place the black cell with that number.
Add a black cell without a number:

2. Press the b key.
Click on the cell where you want to place the black cell.
Add a white cell:

3. Press the w key.
Click on the cell where you want to set it to white. Now, you can change black cells 
(walls) back to white cells. Place a light bulb in edit mode:

4. Press the l key.
Click on the white cell where you want to place or remove a light bulb.
Toggle between playing and editing modes:

5. Press the Esc key to clear your selection and return to playing mode.
In playing mode (no selection), clicking on white cells will place or remove light bulbs.
Check your selection:

The current selection is displayed at the bottom of the grid.

Note:
The initial grid includes some pre-set black cells with numbers as examples. You can customize 
the entire grid using the instructions above.
The GUI updates automatically to reflect your changes, and it checks for violations in real-time.
With this fix, pressing the 'w' key and clicking on any cell (including those with walls) 
will now correctly change it to a white cell.


"""
import tkinter as tk
from tkinter import filedialog
import json
import threading

class LightUpPuzzle:
    def __init__(self, master, grid_size=7):
        self.master = master
        self.grid_size = grid_size
        self.cell_size = 40
        self.grid = []
        self.cells = {}
        self.selected_number = None  # To keep track of the selected number or cell type
        self.selected_method = "bk"  # Default solving method

        # Add dropdown menu for grid size selection
        self.grid_size_var = tk.StringVar(value=str(self.grid_size))
        self.grid_size_dropdown = tk.OptionMenu(self.master, self.grid_size_var, "7", "10", "14", command=self.change_grid_size)
        self.grid_size_dropdown.grid(row=0, column=0, columnspan=2, sticky='w')

        # Updated Save and Load buttons
        self.save_button = tk.Button(self.master, text="Save", command=self.save_puzzle)
        self.save_button.grid(row=0, column=2, columnspan=1, sticky='w')
        self.load_button = tk.Button(self.master, text="Load", command=self.load_puzzle)
        self.load_button.grid(row=0, column=3, columnspan=1, sticky='w')
        
        # Add dropdown menu for method selection
        self.method_var  = tk.StringVar(value="Solver")
        self.method_dropdown = tk.OptionMenu(self.master, self.method_var, "bk", "ac3", command=self.change_method)
        self.method_dropdown.grid(row=0, column=4, columnspan=2, sticky='w')

        self.solve_button = tk.Button(self.master, text="Solve", command=self.solve_puzzle)
        self.solve_button.grid(row=0, column=6, sticky='w')

        self.status_label = tk.Label(self.master, text="Playing mode")
        self.status_label.grid(row=1, column=0, columnspan=self.grid_size)

        self.init_grid()
        self.draw_grid()
        self.master.bind('<Key>', self.on_key_press)  # Bind key events

    def change_grid_size(self, value):
        # Update grid size
        self.grid_size = int(value)
        # Clear existing grid and cells
        for widget in self.master.winfo_children():
            if widget not in [
                self.grid_size_dropdown,
                self.method_dropdown,
                self.status_label,
                self.save_button,
                self.load_button,
            ]:
                widget.destroy()
        self.grid = []
        self.cells = {}
        self.init_grid()
        self.draw_grid()
        self.status_label.config(text="Playing mode")

    def change_method(self, value):
        # Update the selected solving method
        self.selected_method = value
        self.status_label.config(text=f"Selected method: {self.selected_method}")

    def init_grid(self):
        # Initialize grid with empty cells
        self.grid = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                cell = {
                    'type': 'white',
                    'number': None,
                    'illuminated': False,
                    'light_bulb': False,
                    'conflict': False
                }
                row.append(cell)
            self.grid.append(row)

        # Example of setting black cells with numbers
        # You can modify this to set up a specific puzzle
        if self.grid_size >= 7:
            self.grid[1][1]['type'] = 'black'
            self.grid[1][1]['number'] = 2
            self.grid[3][4]['type'] = 'black'
            self.grid[3][4]['number'] = 3
            self.grid[5][2]['type'] = 'black'
            self.grid[5][2]['number'] = 0




    def draw_grid(self):
        start_row = 2  # Adjust starting row because of the dropdowns and status label
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                frame = tk.Frame(self.master, width=self.cell_size, height=self.cell_size)
                frame.propagate(False)
                frame.grid(row=i + start_row, column=j)
                self.cells[(i, j)] = frame  # store the frame
                self.bind_cell(frame, i, j)  # Bind click event to the frame
                self.redraw_cell(i, j)  # Draw the cell content

    def bind_cell(self, frame, x, y):
        frame.bind("<Button-1>", lambda event, x=x, y=y: self.on_cell_click(x, y))
        for widget in frame.winfo_children():
            widget.bind("<Button-1>", lambda event, x=x, y=y: self.on_cell_click(x, y))

    def on_key_press(self, event):
        key = event.char
        if key in '01234':
            self.selected_number = int(key)
            self.status_label.config(text=f"Selected number {self.selected_number}")
        elif key.lower() == 'w':
            self.selected_number = 'white'
            self.status_label.config(text="Selected white cell")
        elif key.lower() == 'b':
            self.selected_number = 'black'
            self.status_label.config(text="Selected black cell")
        elif key.lower() == 'l':
            self.selected_number = 'light_bulb'
            self.status_label.config(text="Selected light bulb")
        elif key == '\x1b':  # Esc key
            self.selected_number = None
            self.status_label.config(text="Playing mode")

    def on_cell_click(self, x, y):
        cell = self.grid[x][y]
        if self.selected_number is not None:
            # Editing mode
            if self.selected_number == 'white':
                # Set cell to white
                cell['type'] = 'white'
                cell['number'] = None
                cell['light_bulb'] = False
            elif self.selected_number == 'black':
                # Set cell to black without number
                cell['type'] = 'black'
                cell['number'] = None
                cell['light_bulb'] = False
            elif self.selected_number == 'light_bulb':
                # Place a light bulb
                if cell['type'] == 'white':
                    cell['light_bulb'] = not cell['light_bulb']
            else:
                # self.selected_number is an integer between 0 and 4
                cell['type'] = 'black'
                cell['number'] = self.selected_number
                cell['light_bulb'] = False
            self.redraw_cell(x, y)
            self.update_illumination()
            self.check_for_violations()
        else:
            # Playing mode
            if cell['type'] == 'white':
                # Toggle light bulb
                cell['light_bulb'] = not cell['light_bulb']
                self.update_illumination()
                self.check_for_violations()

    def redraw_cell(self, x, y):
        cell = self.grid[x][y]
        frame = self.cells[(x, y)]
        # Remove all widgets from frame
        for widget in frame.winfo_children():
            widget.destroy()
        if cell['type'] == 'white':
            btn = tk.Button(frame, bg='white')
            btn.pack(fill=tk.BOTH, expand=True)
            if cell['light_bulb']:
                btn.config(text='💡')
            self.bind_cell(frame, x, y)  # Re-bind click events
        elif cell['type'] == 'black':
            if cell['number'] is not None:
                lbl = tk.Label(frame, text=str(cell['number']), bg='black', fg='white')
            else:
                lbl = tk.Label(frame, bg='black')
            lbl.pack(fill=tk.BOTH, expand=True)
            self.bind_cell(frame, x, y)  # Re-bind click events
        self.cells[(x, y)] = frame

    def update_illumination(self):
        # Clear illumination
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j]['type'] == 'white':
                    self.grid[i][j]['illuminated'] = False
                    self.grid[i][j]['conflict'] = False
        # Set illumination
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j]['light_bulb']:
                    self.illuminate(i, j)
        # Update GUI
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i][j]
                if cell['type'] == 'white':
                    btn = self.cells[(i, j)].winfo_children()[0]
                    if cell['light_bulb']:
                        btn.config(text='💡')
                    else:
                        btn.config(text='')
                    if cell['conflict']:
                        btn.config(bg='red')
                    elif cell['light_bulb']:
                        btn.config(bg='yellow')
                    elif cell['illuminated']:
                        btn.config(bg='yellow')
                    else:
                        btn.config(bg='white')
                elif cell['type'] == 'black' and cell['number'] is not None:
                    lbl = self.cells[(i, j)].winfo_children()[0]
                    if cell['conflict']:
                        lbl.config(fg='red')
                    else:
                        lbl.config(fg='white')

    def illuminate(self, x, y):
        # Illuminate the bulb's own cell
        self.grid[x][y]['illuminated'] = True
        # Illuminate cells in all four directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell = self.grid[nx][ny]
                    if cell['type'] == 'black':
                        break
                    cell['illuminated'] = True
                else:
                    break

    def check_for_violations(self):
        # Reset conflicts
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid[i][j]['conflict'] = False

        # Check for bulb conflicts
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j]['light_bulb']:
                    self.check_bulb_conflicts(i, j)

        # Check numbered black cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i][j]
                if cell['type'] == 'black' and cell['number'] is not None:
                    self.check_numbered_cell(i, j)

        # Update GUI for conflicts and numbered cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i][j]
                frame = self.cells[(i, j)]
                if cell['type'] == 'white':
                    btn = frame.winfo_children()[0]
                    if cell['conflict']:
                        btn.config(bg='red')
                    elif cell['light_bulb']:
                        btn.config(bg='yellow')
                    elif cell['illuminated']:
                        btn.config(bg='yellow')
                    else:
                        btn.config(bg='white')
                elif cell['type'] == 'black' and cell['number'] is not None:
                    lbl = frame.winfo_children()[0]
                    if cell['conflict']:
                        lbl.config(fg='red')
                    else:
                        lbl.config(fg='white')

    def check_bulb_conflicts(self, x, y):
        # Check in all four directions for another bulb
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell = self.grid[nx][ny]
                    if cell['type'] == 'black':
                        break
                    if cell['light_bulb']:
                        # Conflict detected
                        self.grid[x][y]['conflict'] = True
                        self.grid[nx][ny]['conflict'] = True
                else:
                    break

    def check_numbered_cell(self, x, y):
        # Count adjacent bulbs
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        bulb_count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.grid[nx][ny]['light_bulb']:
                    bulb_count += 1
        # Check if the bulb count matches the number
        if bulb_count != self.grid[x][y]['number']:
            self.grid[x][y]['conflict'] = True
        else:
            self.grid[x][y]['conflict'] = False

    def save_puzzle(self):
        # Open file dialog to select save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Puzzle"
        )
        if file_path:
            # Prepare data to save
            data = {
                'grid_size': self.grid_size,
                'grid': self.grid
            }
            # Convert grid data to a serializable format
            serializable_grid = []
            for row in self.grid:
                serializable_row = []
                for cell in row:
                    serializable_cell = cell.copy()
                    # Remove GUI-related keys
                    del serializable_cell['illuminated']
                    del serializable_cell['conflict']
                    serializable_row.append(serializable_cell)
                serializable_grid.append(serializable_row)
            data['grid'] = serializable_grid
            # Save data to file in JSON format
            with open(file_path, 'w') as f:
                json.dump(data, f)
            self.status_label.config(text="Puzzle saved successfully.")

    def load_puzzle(self):
        # Open file dialog to select file to load
        file_path = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Load Puzzle"
        )
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Update grid size if necessary
            new_grid_size = data['grid_size']
            if new_grid_size != self.grid_size:
                self.grid_size = new_grid_size
                self.grid_size_var.set(str(self.grid_size))
                # Clear existing grid and cells
                for widget in self.master.winfo_children():
                    if widget not in [
                        self.grid_size_dropdown,
                        self.method_dropdown,
                        self.status_label,
                        self.save_button,
                        self.load_button,
                    ]:
                        widget.destroy()
                    self.cells = {}
                self.draw_grid()
            # Load grid data
            self.grid = data['grid']
            # Re-initialize missing keys in each cell
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell = self.grid[i][j]
                    cell['illuminated'] = False
                    cell['conflict'] = False
            # Redraw all cells
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.redraw_cell(i, j)
            self.update_illumination()
            self.check_for_violations()
            self.status_label.config(text="Puzzle loaded successfully.")


    def solve_puzzle(self):
        self.status_label.config(text="Solving...")
        thread = threading.Thread(target=self.solve_process)
        thread.start()

    def update_solve_status(self, solved):
        if solved:
            self.status_label.config(text="Puzzle solved!")
            self.update_illumination()
        else:
            self.status_label.config(text="Could not solve the puzzle.")

    def solve_process(self):
        if self.selected_method == 'bk':
            solved = self.solve_using_backtracking()
        elif self.selected_method == 'ac3':
            solved = self.solve_using_ac3()
        else:
            solved = False
        self.master.after(0, self.update_solve_status, solved)

    def solve_using_backtracking(self):
        # Implement your backtracking algorithm here
        return self.backtrack_solve()  # or False based on the success of the solving process

    def backtrack_solve(self, x=0, y=0):
        print(f"Backtracking at {x}, {y}")
        if self.check_all_illuminated():
            return True

        if x == self.grid_size:
            return False

        next_x, next_y = (x + 1, 0) if y == self.grid_size - 1 else (x, y + 1)
        # Skip if it's a black cell or already illuminated.
        if self.grid[x][y]['type'] == 'black' or self.grid[x][y]['light_bulb']:
            print(f"Skipping black cell or already illuminated at {x}, {y}")
            return self.backtrack_solve(next_x, next_y)
        
        self.grid[x][y]['light_bulb'] = True
        self.update_illumination()

        if self.is_valid_placement(x, y) and self.backtrack_solve(next_x, next_y):
            # Try to place a bulb here.
            print(f"Placing bulb at {x}, {y}")
            # If valid, continue to the next cell.
            print(f"Valid placement at {x}, {y}")
            return True
        # Remove bulb if it's not valid.
        print(f"Invalid placement at {x}, {y}")
        self.grid[x][y]['light_bulb'] = False
        self.update_illumination()
        return self.backtrack_solve(next_x, next_y)

    def solve_using_ac3(self):
        # Initialize domains for AC3.
        self.domains = {(i, j): True for i in range(len(self.grid)) for j in range(len(self.grid[0])) if self.grid[i][j]['type'] == 'white'}
        #print(f"solve_using_ac3 Initializing domains for AC3: {self.domains}, {len(self.grid)}, {len(self.grid[0])}")
        solved = False
        queue = []
        # Find the first possible cell to place a bulb.
        for cell in self.domains:
            if self.domains[cell]:
                queue.append(cell)
                break
        else:
            return False
        while queue:
            # 1. Acquire a cell to place a bulb.
            x, y = queue[-1]
            print(f"Placing bulb at {x}, {y}")
            self.grid[x][y]['light_bulb'] = True
            self.update_illumination()
            if self.is_valid_placement(x, y):
                # 2. Check if the puzzle is solved.
                if self.check_all_illuminated():
                    solved = True
                    break
                # 3. Apply constraints and find the next cell to place a bulb.
                affected = self.apply_constraints(x, y)
                print(f"Affected cells: {affected}")
            else:
                self.grid[x][y]['light_bulb'] = False
                self.update_illumination()
            next_cell = None
            # 4. Find the next cell to place a bulb.
            cells = list(self.domains.keys())
            current_index = cells.index((x, y)) + 1
            for index in range(current_index, len(cells)):
                cell = cells[index]
                if self.domains[cell]:
                    next_cell = cell
                    break
            print(f"Find Next cell: current_index={current_index}, next_cell={next_cell}")
            # 5. If no cell is found, rollback changes and try again.
            if next_cell is None:
                self.grid[x][y]['light_bulb'] = False
                self.update_illumination()
                self.rollback_changes(affected)
                self.update_illumination()
                cell_to_remove_x, cell_to_remove_y = queue.pop()
                self.domains[(cell_to_remove_x, cell_to_remove_y)] = False
                print(f"Rollback changes and remove cell from queue: {cell_to_remove_x}, {cell_to_remove_y}")
                continue
            # 6. Add the next cell to the queue.
            queue.append(next_cell)

        return solved


    def apply_constraints(self, x, y):
        print(f"Applying constraints at {x}, {y}")
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        affected_cells = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                if self.grid[nx][ny]['type'] == 'white' and self.domains[(nx, ny)]:
                    self.domains[(nx, ny)] = False
                    affected_cells.append((nx, ny))
                    print(f"Affected cell by direct conflict: {nx}, {ny}")
                elif self.grid[nx][ny]['type'] == 'black':
                    # If the number of bulbs exceeds the limit, disable the bulbs in the adjacent cells.
                    if 'number' in self.grid[nx][ny] and not self.can_place_more_bulbs(nx, ny):
                        for adj_dx, adj_dy in directions:
                            adj_nx, adj_ny = nx + adj_dx, ny + adj_dy
                            if 0 <= adj_nx < len(self.grid) and 0 <= adj_ny < len(self.grid[0]):
                                if self.grid[adj_nx][adj_ny]['type'] == 'white' and self.domains[(adj_nx, adj_ny)]:
                                    self.domains[(adj_nx, adj_ny)] = False
                                    affected_cells.append((adj_nx, adj_ny))
                                    print(f"Affected cell by number constraint: {adj_nx}, {adj_ny}")
                    break
                nx += dx
                ny += dy
        return affected_cells

    def rollback_changes(self, changes):
        for cell in changes:
            self.domains[cell] = True


    def is_valid_placement(self, x, y):
        return self.check_direct_conflicts(x, y) and self.check_immediate_numbers(x, y)

    # Check if there are any bulbs in the same row or column.
    def check_direct_conflicts(self, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.grid[nx][ny]['type'] == 'black':
                    break
                if self.grid[nx][ny]['light_bulb']:
                    return False
                nx += dx
                ny += dy
        return True

    def check_immediate_numbers(self, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[nx][ny]['type'] == 'black' and 'number' in self.grid[nx][ny]:
                if not self.can_place_more_bulbs(nx, ny):
                    return False
        return True

    def can_place_more_bulbs(self, x, y):
        required_bulbs = self.grid[x][y]['number']
        print(f"can_place_more_bulbs Required bulbs: {required_bulbs}")
        adjacent_cells = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        bulb_count = sum(1 for ax, ay in adjacent_cells if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size and self.grid[ax][ay]['light_bulb'])
        # Only return False if the count exceeds what is allowed, not merely for being unequal
        return bulb_count <= required_bulbs

    def check_all_illuminated(self):
        print("Checking all illuminated")
        # Check if all white cells are illuminated.
        return all(self.grid[x][y]['illuminated'] for x in range(self.grid_size) for y in range(self.grid_size) if self.grid[x][y]['type'] == 'white')


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Light Up Puzzle")
    app = LightUpPuzzle(root)
    root.mainloop()


