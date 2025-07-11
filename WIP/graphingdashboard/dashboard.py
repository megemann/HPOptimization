import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
import glob
from pathlib import Path

# Initialize the Dash app
app = dash.Dash(__name__)

class ResultsPlotter:
    def __init__(self, base_path=".."):
        self.base_path = Path(base_path)
        self.colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5"
        ]
    
    def find_result_directories(self):
        """Find all directories containing result files"""
        result_dirs = []
        
        # Search for different result file patterns
        patterns = [
            "**/results_*.json",  # CaliforniaHousing/UCILetter format
            "**/*_results.json",  # Branin format
        ]
        
        for pattern in patterns:
            for file_path in self.base_path.glob(pattern):
                dir_path = file_path.parent
                if dir_path not in result_dirs:
                    result_dirs.append(dir_path)
        
        return sorted(result_dirs)
    
    def detect_data_format(self, directory):
        """Detect which data format this directory uses"""
        dir_path = Path(directory)
        
        # Check for different file patterns
        if list(dir_path.glob("results_*.json")):
            return "phase2"  # CaliforniaHousing/UCILetter format
        elif list(dir_path.glob("*_results.json")):
            return "branin"  # Branin format
        else:
            return "unknown"
    
    def detect_available_metrics(self, directory):
        """Detect which metrics are available in a directory"""
        data_format = self.detect_data_format(directory)
        available_metrics = set()
        
        if data_format == "phase2":
            results_list = self.load_phase2_results(directory)
            if results_list:
                for results in results_list:
                    if len(results) > 1:
                        # Skip start_time entry if present
                        result_entries = results[1:] if isinstance(results[0], dict) and "start_time" in results[0] else results
                        
                        for entry in result_entries:
                            if isinstance(entry, dict):
                                if "accuracy" in entry:
                                    available_metrics.add("accuracy")
                                if "loss" in entry:
                                    available_metrics.add("loss")
                                if "mse" in entry:
                                    available_metrics.add("mse")
                            
                            # Only check first few entries to avoid processing all data
                            if len(available_metrics) >= 3:
                                break
                        
                        if len(available_metrics) >= 2:
                            break
        
        return list(available_metrics)
    
    def load_phase2_results(self, directory):
        """Load Phase 2 format results (CaliforniaHousing/UCILetter)"""
        results_list = []
        dir_path = Path(directory)
        
        for i in range(1, 4):  # Try results_1.json, results_2.json, results_3.json
            file_path = dir_path / f"results_{i}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        results_list.append(json.load(f))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return results_list
    
    def load_branin_results(self, directory):
        """Load Branin format results"""
        dir_path = Path(directory)
        
        # Look for files like 50_results.json, 2_results.json, etc.
        result_files = list(dir_path.glob("*_results.json"))
        if result_files:
            try:
                with open(result_files[0], "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {result_files[0]}: {e}")
        
        return None
    
    def extract_phase2_time_series(self, results, metric_type="auto"):
        """Extract time series from Phase 2 format
        
        Args:
            results: The results data
            metric_type: "auto", "accuracy", "loss", or "mse" - which metric to extract
        """
        if len(results) <= 1:
            return None, None
        
        # Handle start time
        if isinstance(results[0], dict) and "start_time" in results[0]:
            start_time = results[0]["start_time"]
            result_entries = results[1:]
        else:
            start_time = min([r.get("trial_end", 0) for r in results if "trial_end" in r], default=0)
            result_entries = results
        
        times = []
        values = []
        best_value_so_far = None
        
        for entry in result_entries:
            value = None
            
            # Extract the requested metric
            if metric_type == "accuracy" and "accuracy" in entry:
                value = entry["accuracy"]
                if best_value_so_far is None:
                    best_value_so_far = value
                else:
                    best_value_so_far = max(best_value_so_far, value)
            elif metric_type == "loss" and "loss" in entry:
                value = entry["loss"]
                if best_value_so_far is None:
                    best_value_so_far = value
                else:
                    best_value_so_far = min(best_value_so_far, value)
            elif metric_type == "mse" and "mse" in entry:
                value = entry["mse"]
                if best_value_so_far is None:
                    best_value_so_far = value
                else:
                    best_value_so_far = min(best_value_so_far, value)
            elif metric_type == "auto":
                # Auto-detect metric (original behavior)
                if "mse" in entry:
                    value = entry["mse"]
                    if best_value_so_far is None:
                        best_value_so_far = value
                    else:
                        best_value_so_far = min(best_value_so_far, value)
                elif "accuracy" in entry:
                    value = entry["accuracy"]
                    if best_value_so_far is None:
                        best_value_so_far = value
                    else:
                        best_value_so_far = max(best_value_so_far, value)
                elif "loss" in entry:
                    value = entry["loss"]
                    if best_value_so_far is None:
                        best_value_so_far = value
                    else:
                        best_value_so_far = min(best_value_so_far, value)
            
            if value is not None and "trial_end" in entry:
                rel_time = entry["trial_end"] - start_time
                times.append(rel_time)
                values.append(best_value_so_far)
        
        if times:
            sorted_data = sorted(zip(times, values))
            times, values = zip(*sorted_data)
            return times, values
        
        return None, None
    
    def extract_branin_time_series(self, results, method_name="random_results", k=None):
        """Extract time series from Branin format"""
        if method_name not in results:
            return None, None
        
        method_runs = results[method_name]
        if k is None:
            k = len(method_runs)
        
        per_run_series = []
        t_max = 0.0
        
        for run in method_runs[:k]:
            trials = run["trials"]
            if not trials:
                continue
                
            t0 = datetime.fromisoformat(trials[0]["timestamp"])
            
            best_so_far = np.inf
            times, bests = [], []
            
            for trial in trials:
                t_sec = (datetime.fromisoformat(trial["timestamp"]) - t0).total_seconds()
                best_so_far = min(best_so_far, trial["value"])
                times.append(t_sec)
                bests.append(best_so_far)
            
            if times:
                s = pd.Series(bests, index=times)
                s = s.groupby(level=0).last().sort_index()
                per_run_series.append(s)
                t_max = max(t_max, s.index[-1])
        
        if not per_run_series:
            return None, None
        
        # Create uniform grid and average
        bin_size = 0.1
        grid = np.arange(0, t_max + bin_size, bin_size)
        
        filled = []
        for s in per_run_series:
            s = s.reindex(sorted(set(grid).union(s.index))).sort_index()
            s = s.ffill()
            filled.append(s.reindex(grid))
        
        if filled:
            avg_best = pd.concat(filled, axis=1).mean(axis=1)
            return grid, avg_best.values
        
        return None, None
    
    def process_directory(self, directory, metric_type="auto"):
        """Process a directory and return time series data
        
        Args:
            directory: Directory path
            metric_type: "auto", "accuracy", "loss", or "mse" - which metric to extract
        """
        data_format = self.detect_data_format(directory)
        
        if data_format == "phase2":
            results_list = self.load_phase2_results(directory)
            if not results_list:
                return None
            
            # Process each run and average if multiple runs
            time_series_data = []
            
            for results in results_list:
                times, values = self.extract_phase2_time_series(results, metric_type)
                if times and values:
                    time_series_data.append((times, values))
            
            if not time_series_data:
                return None
            
            if len(time_series_data) > 1:
                # Average across runs and calculate standard deviation
                max_time_data = max([max(times) for times, _ in time_series_data])
                ref_times = np.linspace(0, max_time_data, 200)
                
                interp_values = []
                for times, values in time_series_data:
                    if len(times) > 1:
                        value_interp = interp1d(times, values, bounds_error=False, 
                                              fill_value=(values[0], values[-1]))
                        interp_values.append(value_interp(ref_times))
                
                if interp_values:
                    interp_values = np.array(interp_values)
                    avg_values = np.mean(interp_values, axis=0)
                    std_values = np.std(interp_values, axis=0)
                    return {"times": ref_times, "values": avg_values, "std": std_values, "runs": len(results_list), "metric": metric_type}
            else:
                times, values = time_series_data[0]
                return {"times": times, "values": values, "runs": 1, "metric": metric_type}
        
        elif data_format == "branin":
            results = self.load_branin_results(directory)
            if not results:
                return None
            
            # Try different method names
            method_names = ["random_results", "tpe_results", "gp_results", 
                          "cmaes_results", "hyperband_results", "grid_results"]
            
            for method_name in method_names:
                times, values = self.extract_branin_time_series(results, method_name)
                if times is not None and values is not None:
                    return {"times": times, "values": values, "method": method_name}
        
        return None

# Create the plotter instance
plotter = ResultsPlotter()

# Get available directories
available_dirs = plotter.find_result_directories()
dir_options = [{"label": str(d.relative_to(plotter.base_path)), "value": str(d)} 
               for d in available_dirs]

# Create tree structure for directories
def build_tree_structure():
    tree = {}
    
    # First, collect all the actual result directories as sets for fast lookup
    result_dirs_set = set(str(d) for d in available_dirs)
    
    # Build all possible paths first to ensure consistent structure
    all_paths = set()
    for d in available_dirs:
        rel_path = str(d.relative_to(plotter.base_path))
        parts = rel_path.split(os.sep)
        
        # Add all parent paths
        for i in range(1, len(parts) + 1):
            path_so_far = os.sep.join(parts[:i])
            full_path = str(plotter.base_path / path_so_far)
            all_paths.add((path_so_far, full_path))
    
    # Now build the tree with all paths
    for rel_path, full_path in all_paths:
        parts = rel_path.split(os.sep)
        current = tree
        
        for i, part in enumerate(parts):
            if part not in current:
                # Determine if this is a result directory
                current_full_path = str(plotter.base_path / os.sep.join(parts[:i+1]))
                is_result_dir = current_full_path in result_dirs_set
                
                current[part] = {
                    '_children': {},
                    '_is_result_dir': is_result_dir,
                    '_full_path': current_full_path if is_result_dir else None,
                    '_rel_path': os.sep.join(parts[:i+1])
                }
            current = current[part]['_children']
    
    return tree

tree_structure = build_tree_structure()

# Function to sanitize IDs for Dash
def sanitize_id(text):
    """Replace invalid characters in Dash component IDs"""
    return text.replace('.', '_DOT_').replace('=', '_EQ_').replace(' ', '_SPACE_').replace('/', '_SLASH_')

# Create flat list for callback management
all_node_ids = []
def collect_node_ids(node, prefix=""):
    global all_node_ids
    for key, value in node.items():
        if not key.startswith('_'):
            node_id = f"{prefix}-{key}" if prefix else key
            sanitized_id = sanitize_id(node_id)
            all_node_ids.append(sanitized_id)
            collect_node_ids(value['_children'], node_id)

collect_node_ids(tree_structure)

# Function to create tree UI
def create_tree_node(node_dict, node_key, node_id, level=0):
    """Create a tree node with expand/collapse functionality"""
    node_info = node_dict[node_key]
    children = node_info['_children']
    is_result_dir = node_info['_is_result_dir']
    
    indent = 20 * level
    sanitized_node_id = sanitize_id(node_id)
    
    # Create child nodes
    child_nodes = []
    for child_key in sorted(children.keys()):
        child_id = f"{node_id}-{child_key}" if node_id else child_key
        child_nodes.append(create_tree_node(children, child_key, child_id, level + 1))
    
    # If this is a result directory, show checkbox
    if is_result_dir:
        checkbox_element = dcc.Checklist(
            id=f'leaf-{sanitized_node_id}',
            options=[{'label': f" {node_key}", 'value': node_info['_full_path']}],
            value=[],
            labelStyle={'display': 'inline', 'margin': '0', 'color': '#28a745', 'fontWeight': 'bold'},
            inputStyle={'margin': '0 5px 0 0'}
        )
        icon = "📊"  # Chart icon for result directories
        color = '#28a745'  # Green for result directories
    else:
        checkbox_element = None
        icon = "📁"  # Folder icon for regular directories
        color = '#495057'  # Regular color for folders
    
    # Create the main node content
    node_content = []
    
    # If has children, create expandable folder
    if child_nodes:
        node_content.append(
            html.Div([
                html.Button(
                    [html.Span("▶", id=f'arrow-{sanitized_node_id}', style={'marginRight': '5px'}),
                     html.Span(icon, style={'marginRight': '5px'}),
                     node_key,
                     html.Span(" " if checkbox_element else "", style={'marginLeft': '10px'}),
                     checkbox_element if checkbox_element else html.Span()],
                    id=f'folder-{sanitized_node_id}',
                    n_clicks=0,
                    style={
                        'background': 'none',
                        'border': 'none',
                        'padding': '2px 5px',
                        'cursor': 'pointer',
                        'fontSize': '14px',
                        'color': color,
                        'fontWeight': 'bold' if is_result_dir else 'normal',
                        'width': '100%',
                        'textAlign': 'left'
                    }
                )
            ], style={'marginLeft': f'{indent}px'}),
        )
        
        node_content.append(
            html.Div(
                child_nodes,
                id=f'children-{sanitized_node_id}',
                style={'display': 'none'}  # Initially collapsed
            )
        )
    else:
        # Leaf node without children - just show the checkbox
        if is_result_dir:
            node_content.append(
                html.Div([
                    html.Span(icon, style={'marginRight': '5px', 'color': color}),
                    checkbox_element
                ], style={'marginLeft': f'{indent}px', 'padding': '2px 0'})
            )
    
    return html.Div(node_content)

def create_tree_ui():
    """Create the complete tree UI"""
    tree_nodes = []
    for root_key in sorted(tree_structure.keys()):
        tree_nodes.append(create_tree_node(tree_structure, root_key, root_key))
    
    return html.Div(tree_nodes, style={
        'maxHeight': '400px', 
        'overflowY': 'auto', 
        'border': '1px solid #dee2e6', 
        'padding': '10px',
        'backgroundColor': '#f8f9fa'
    })

# App layout
app.layout = html.Div([
    html.H1("Optimization Results Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Select Directories to Plot:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            html.Br(),
            create_tree_ui()
        ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Plot Title:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
            dcc.Input(
                id='plot-title',
                type='text',
                value='Optimization Results',
                placeholder='Enter plot title...',
                style={'width': '100%', 'padding': '5px'}
            ),
            html.Br(), html.Br(),
            html.Label("Selected Directories:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
            html.Div(id='selected-count', style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6', 'borderRadius': '3px', 'minHeight': '100px'})
        ], style={'width': '38%', 'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    
    html.Br(),
    
    html.Div([
        html.Div([
            html.Button('Clear All', id='clear-all-button', n_clicks=0, 
                       style={'margin': '5px', 'backgroundColor': '#dc3545', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'borderRadius': '3px'}),
            html.Button('Select All', id='select-all-button', n_clicks=0,
                       style={'margin': '5px', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'borderRadius': '3px'})
        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
        
        html.Div([
            html.Label("Y-axis Label:"),
            dcc.Input(id='y-axis-label', type='text', placeholder='Auto', value='Best Value')
        ], style={'width': '18%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Y Min:"),
            dcc.Input(id='y-min', type='number', placeholder='Auto')
        ], style={'width': '15%', 'display': 'inline-block', 'marginLeft': '2%'}),
        
        html.Div([
            html.Label("Y Max:"),
            dcc.Input(id='y-max', type='number', placeholder='Auto')
        ], style={'width': '15%', 'display': 'inline-block', 'marginLeft': '2%'}),
        
        html.Div([
            html.Label("X Max (Time):"),
            dcc.Input(id='x-max', type='number', placeholder='Auto')
        ], style={'width': '15%', 'display': 'inline-block', 'marginLeft': '2%'}),
        
        html.Div([
            html.Button('Update Plot', id='update-button', n_clicks=0,
                       style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '8px 15px', 'borderRadius': '3px', 'fontWeight': 'bold'})
        ], style={'width': '18%', 'display': 'inline-block', 'marginLeft': '2%', 'marginTop': '25px'})
    ]),
    
    html.Br(),
    html.Br(),
    
    dcc.Graph(id='results-graph'),
    
    html.Div(id='plot-info', style={'marginTop': '20px'})
])

# Get all leaf node IDs for callbacks
def get_all_leaf_ids():
    """Get IDs only for directories that actually contain result files and are true leaf nodes"""
    leaf_ids = []
    
    # Only include directories that are in our available_dirs list
    available_paths = set(str(d) for d in available_dirs)
    
    def find_result_dirs(node, prefix=""):
        for key, value in node.items():
            if not key.startswith('_'):
                node_id = f"{prefix}-{key}" if prefix else key
                
                # Only add if this is a result directory AND has no children (true leaf)
                # OR if it's a result directory that has children but they're not result directories
                if value['_is_result_dir'] and value['_full_path'] in available_paths:
                    # Check if this node has any children that are also result directories
                    has_result_children = any(
                        child_info['_is_result_dir'] 
                        for child_info in value['_children'].values()
                    )
                    
                    # Only add this node if it doesn't have result directory children
                    # (i.e., it's a true leaf or all its children are non-result folders)
                    if not has_result_children:
                        sanitized_id = sanitize_id(node_id)
                        leaf_ids.append(f'leaf-{sanitized_id}')
                
                # Always recurse through children to find nested result directories
                find_result_dirs(value['_children'], node_id)
    
    find_result_dirs(tree_structure)
    return leaf_ids

all_leaf_ids = get_all_leaf_ids()

# Callback for folder expand/collapse
def create_folder_callbacks():
    def find_folder_ids(node, prefix=""):
        folder_ids = []
        for key, value in node.items():
            if not key.startswith('_'):
                node_id = f"{prefix}-{key}" if prefix else key
                # If this node has children, it's a folder (regardless of whether it's also a result dir)
                if value['_children']:
                    sanitized_id = sanitize_id(node_id)
                    folder_ids.append(sanitized_id)
                    folder_ids.extend(find_folder_ids(value['_children'], node_id))
        return folder_ids
    
    folder_ids = find_folder_ids(tree_structure)
    
    for folder_id in folder_ids:
        @app.callback(
            [Output(f'children-{folder_id}', 'style'),
             Output(f'arrow-{folder_id}', 'children')],
            [Input(f'folder-{folder_id}', 'n_clicks')],
            prevent_initial_call=True
        )
        def toggle_folder(n_clicks, fid=folder_id):
            if n_clicks % 2 == 1:  # Expanded
                return {'display': 'block'}, '▼'
            else:  # Collapsed
                return {'display': 'none'}, '▶'

create_folder_callbacks()

# Callbacks for select all and clear all buttons
@app.callback(
    [Output(leaf_id, 'value') for leaf_id in all_leaf_ids],
    [Input('clear-all-button', 'n_clicks'), Input('select-all-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_all_checkboxes(clear_clicks, select_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return [[] for _ in all_leaf_ids]
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'clear-all-button':
        return [[] for _ in all_leaf_ids]
    elif button_id == 'select-all-button':
        # Get the single option for each leaf by finding the corresponding directory
        result = []
        
        # Create a mapping from leaf_id to full_path
        leaf_id_to_path = {}
        
        def map_leaf_ids(node, prefix=""):
            for key, value in node.items():
                if not key.startswith('_'):
                    node_id = f"{prefix}-{key}" if prefix else key
                    
                    if value['_is_result_dir'] and value['_full_path']:
                        # Check if this should be included (same logic as get_all_leaf_ids)
                        available_paths = set(str(d) for d in available_dirs)
                        if value['_full_path'] in available_paths:
                            has_result_children = any(
                                child_info['_is_result_dir'] 
                                for child_info in value['_children'].values()
                            )
                            
                            if not has_result_children:
                                sanitized_id = sanitize_id(node_id)
                                leaf_id_to_path[f'leaf-{sanitized_id}'] = value['_full_path']
                    
                    map_leaf_ids(value['_children'], node_id)
        
        map_leaf_ids(tree_structure)
        
        # Now create the result list
        for leaf_id in all_leaf_ids:
            if leaf_id in leaf_id_to_path:
                result.append([leaf_id_to_path[leaf_id]])
            else:
                result.append([])
        
        return result
    
    return [[] for _ in all_leaf_ids]

# Callback to update selected directories display
@app.callback(
    Output('selected-count', 'children'),
    [Input(leaf_id, 'value') for leaf_id in all_leaf_ids]
)
def update_selected_display(*checklist_values):
    selected_dirs = []
    for values in checklist_values:
        if values:
            selected_dirs.extend(values)
    
    if not selected_dirs:
        return html.Div("No directories selected", style={'color': '#6c757d', 'fontStyle': 'italic'})
    
    selected_labels = []
    for dir_path in selected_dirs:
        rel_path = str(Path(dir_path).relative_to(plotter.base_path))
        selected_labels.append(html.Div(f"• {rel_path}", style={'margin': '2px 0', 'fontSize': '12px'}))
    
    return [
        html.Div(f"Selected: {len(selected_dirs)} directories", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
        html.Div(selected_labels)
    ]

@app.callback(
    [Output('results-graph', 'figure'),
     Output('plot-info', 'children')],
    [Input('update-button', 'n_clicks')],
    [State(leaf_id, 'value') for leaf_id in all_leaf_ids] +
    [State('plot-title', 'value'),
     State('y-axis-label', 'value'),
     State('y-min', 'value'),
     State('y-max', 'value'),
     State('x-max', 'value')]
)
def update_graph(n_clicks, *args):
    # Extract checklist values and other parameters
    num_leaves = len(all_leaf_ids)
    checklist_values = args[:num_leaves]
    plot_title, y_axis_label, y_min, y_max, x_max = args[num_leaves:]
    
    # Combine all selected directories from all checklists
    selected_dirs = []
    for values in checklist_values:
        if values:
            selected_dirs.extend(values)
    
    if not selected_dirs:
        fig = go.Figure()
        fig.update_layout(title="No directories selected")
        return fig, "Select directories to plot results"
    
    fig = go.Figure()
    plot_info = []
    
    color_idx = 0
    for directory in selected_dirs:
        # Check what metrics are available for this directory
        available_metrics = plotter.detect_available_metrics(directory)
        
        # If both accuracy and loss are available, plot both
        if "accuracy" in available_metrics and "loss" in available_metrics:
            metrics_to_plot = ["accuracy", "loss"]
        else:
            metrics_to_plot = ["auto"]  # Use auto-detection for single metric
        
        for metric in metrics_to_plot:
            data = plotter.process_directory(directory, metric)
            if data is None:
                continue
            
            # Generate line name from directory path - just use the final directory name
            dir_path = Path(directory)
            relative_path = dir_path.relative_to(plotter.base_path)
            final_dir = relative_path.parts[-1]  # Just the last directory name
            
            # Add metric suffix if plotting multiple metrics
            if len(metrics_to_plot) > 1:
                line_name = f"{final_dir} ({metric})"
            else:
                line_name = final_dir
            
            times = data["times"]
            values = data["values"]
            
            # Apply time filter if specified
            if x_max is not None:
                mask = np.array(times) <= x_max
                times = np.array(times)[mask]
                values = np.array(values)[mask]
            
            # Use the line name as is (no run count in legend)
            if "method" in data:
                line_label = f"{line_name} ({data['method']})"
            else:
                line_label = line_name
            
            # Add trace with markers and alternating line styles
            color = plotter.colors[color_idx % len(plotter.colors)]
            
            # Define different markers
            markers = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 
                      'star', 'hexagon', 'pentagon', 'cross', 'x']
            marker = markers[color_idx % len(markers)]
            
            # Add main line with markers first
            fig.add_trace(go.Scatter(
                x=times,
                y=values,
                mode='lines+markers',
                name=line_label,
                line=dict(color=color, width=2),
                marker=dict(symbol=marker, size=6, color=color)
            ))
            
            # Add standard deviation band if available (after main line)
            if "std" in data and len(data["std"]) == len(values):
                std_values = data["std"]
                upper_bound = np.array(values) + std_values
                lower_bound = np.array(values) - std_values
                
                # Convert hex color to RGB for rgba
                hex_color = color.lstrip('#')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                
                # Add the confidence band using a single trace
                fig.add_trace(go.Scatter(
                    x=list(times) + list(times[::-1]),  # x, then x reversed
                    y=list(upper_bound) + list(lower_bound[::-1]),  # upper, then lower reversed
                    fill='toself',
                    fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),  # Transparent line
                    showlegend=False,
                    hoverinfo='skip',
                    name=f'{line_label} ±1σ'
                ))
            
            # Collect info for display
            # Determine best value based on metric type
            if metric == "accuracy" or (metric == "auto" and len(values) > 1 and values[-1] > values[0] and max(values) > 0.5):
                best_value = max(values) if len(values) > 0 else "N/A"
            else:
                best_value = min(values) if len(values) > 0 else "N/A"
            
            plot_info.append(f"{line_label}: Best = {best_value:.4f}" if isinstance(best_value, (int, float)) else f"{line_label}: Best = {best_value}")
            
            color_idx += 1
    
    # Update layout
    fig.update_layout(
        title=plot_title or "Optimization Results",
        xaxis_title="Time (seconds)",
        yaxis_title=y_axis_label or "Best Value",
        legend=dict(x=1.05, y=1),
        template="plotly_white",
        height=600
    )
    
    # Set axis limits
    if y_min is not None or y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])
    
    if x_max is not None:
        fig.update_xaxes(range=[0, x_max])
    
    info_text = html.Div([
        html.H4("Plot Information:"),
        html.Ul([html.Li(info) for info in plot_info])
    ])
    
    return fig, info_text

if __name__ == '__main__':
    app.run(debug=True, port=8050) 