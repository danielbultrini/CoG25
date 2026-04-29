import numpy as np
import simple_wfc as dqi_tools  # Your original file

def extract_advanced_rules(image_array, max_stride=2):
    """
    Scans the image for both local (stride=1) and non-local (stride>1) patterns.
    Returns a list of deduced relationships.
    """
    rules = []
    H, W = image_array.shape
    
    print("\n--- Deducing Rules from Image ---")
    
    # Check horizontal strides (dx)
    for d in range(1, max_stride + 1):
        if W > d:
            h_diff = np.abs(image_array[:, :-d] - image_array[:, d:])
            mean_diff = np.mean(h_diff)
            
            if mean_diff >= 0.8:
                rules.append({'dx': d, 'dy': 0, 'parity': 1})
                print(f"Rule: Horizontal distance {d} -> MUST BE DIFFERENT (Parity 1)")
            elif mean_diff <= 0.2:
                rules.append({'dx': d, 'dy': 0, 'parity': 0})
                print(f"Rule: Horizontal distance {d} -> MUST BE SAME (Parity 0)")
                
    # Check vertical strides (dy)
    for d in range(1, max_stride + 1):
        if H > d:
            v_diff = np.abs(image_array[:-d, :] - image_array[d:, :])
            mean_diff = np.mean(v_diff)
            
            if mean_diff >= 0.8:
                rules.append({'dx': 0, 'dy': d, 'parity': 1})
                print(f"Rule: Vertical distance {d} -> MUST BE DIFFERENT (Parity 1)")
            elif mean_diff <= 0.2:
                rules.append({'dx': 0, 'dy': d, 'parity': 0})
                print(f"Rule: Vertical distance {d} -> MUST BE SAME (Parity 0)")
                
    return rules

def generate_advanced_constraints(grid_width, grid_height, rule_config):
    """
    Dynamically builds the B matrix and v vector using the extracted rules
    (both local and non-local) and boundary constraints.
    """
    pattern_rules = rule_config.get('pattern_rules', [])
    boundaries = rule_config.get('boundaries', {})
    
    num_variables = grid_width * grid_height
    equations = []
    v_vector = []
    
    def get_index(x, y):
        return y * grid_width + x

    # 1. Apply Pattern Rules
    for y in range(grid_height):
        for x in range(grid_width):
            current_idx = get_index(x, y)
            
            for r in pattern_rules:
                nx, ny = x + r['dx'], y + r['dy']
                
                # Apply if the target cell is within the chunk boundaries
                if nx < grid_width and ny < grid_height:
                    eq = [0] * num_variables
                    eq[current_idx] = 1
                    eq[get_index(nx, ny)] = 1
                    equations.append(eq)
                    v_vector.append(r['parity'])
                    
    # 2. Inject boundary conditions from already generated chunks
    for (x, y), val in boundaries.items():
        eq = [0] * num_variables
        eq[get_index(x, y)] = 1
        equations.append(eq)
        v_vector.append(val)
        
    return np.array(equations), np.array(v_vector)

# Inject the advanced constraint generator into the original module
dqi_tools.generate_wfc_constraints = generate_advanced_constraints

def generate_large_map(sample_image, total_W, total_H, chunk_W=3, chunk_H=3, overlap=1, max_stride=2):
    """
    Iteratively generates a large map by overlapping smaller DQI chunks,
    utilizing both local and non-local rules.
    """
    # 1. Deduce rules from the sample image
    deduced_rules = extract_advanced_rules(sample_image, max_stride)
    
    # 2. Setup the global map canvas (-1 means unassigned)
    global_map = np.full((total_H, total_W), -1, dtype=int)
    
    # Calculate step sizes based on required overlap
    step_x = chunk_W - overlap
    step_y = chunk_H - overlap
    
    for start_y in range(0, total_H, step_y):
        for start_x in range(0, total_W, step_x):
            
            # Snap to edges to prevent out-of-bounds
            end_x = min(start_x + chunk_W, total_W)
            end_y = min(start_y + chunk_H, total_H)
            current_start_x = max(0, end_x - chunk_W)
            current_start_y = max(0, end_y - chunk_H)
            
            print(f"\nSolving chunk at X:{current_start_x}, Y:{current_start_y}...")
            
            # Identify boundary constraints from already generated sections
            boundaries = {}
            for cy in range(chunk_H):
                for cx in range(chunk_W):
                    gx = current_start_x + cx
                    gy = current_start_y + cy
                    if global_map[gy, gx] != -1:
                        boundaries[(cx, cy)] = global_map[gy, gx]
            
            # Package rules for the injected generator
            rule_config = {
                'pattern_rules': deduced_rules,
                'boundaries': boundaries
            }
            
            # Run the quantum solver (Using GJE to stay under RAM limits)
            chunk_grid, _, _ = dqi_tools.run_quantum_wfc_GJ(chunk_W, chunk_H, rule=rule_config)
            
            # Write the solved chunk into the global map
            for cy in range(chunk_H):
                for cx in range(chunk_W):
                    gx = current_start_x + cx
                    gy = current_start_y + cy
                    global_map[gy, gx] = chunk_grid[cy, cx]
                    
    return global_map

if __name__ == "__main__":
    # A sample image with a 2-pixel repeating checker pattern
    # This will generate both stride=1 (diff) and stride=2 (same) rules
    sample_img = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    
    # Generate a large 6x6 map using smaller 3x3 quantum solutions
    print("Starting Advanced Iterative Quantum WFC...")
    large_map = generate_large_map(
        sample_image=sample_img, 
        total_W=6, 
        total_H=6, 
        chunk_W=3, 
        chunk_H=3, 
        overlap=1,
        max_stride=2
    )
    
    print("\n--- Final Large Procedural Grid ---")
    print(large_map)
    
    print("\n--- Visualized Output ---")
    biome_map = {0: "🟦", 1: "🟩"}
    for row in large_map:
        print("".join([biome_map[cell] for cell in row]))