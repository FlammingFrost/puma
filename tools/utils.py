import os
import re

def gitignore_to_regex(pattern):
    """Convert a .gitignore pattern to a regex pattern."""
    pattern = pattern.strip()
    if not pattern or pattern.startswith('#'):
        return None  # Ignore empty lines and comments
    
    pattern = pattern.replace('.', r'\.')  # Escape dots
    pattern = pattern.replace('*', '.*')  # Convert * to .*
    pattern = pattern.replace('?', '.')  # Convert ? to .
    
    if pattern.endswith('/'):
        pattern = f'^{pattern}.*'
    else:
        pattern = f'^{pattern}$'
    
    return re.compile(pattern)

def load_gitignore(gitignore_path):
    """Load .gitignore patterns and convert them to regex."""
    regex_patterns = []
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        for line in f:
            regex = gitignore_to_regex(line)
            if regex:
                regex_patterns.append(regex)
    return regex_patterns

def is_ignored(path, regex_patterns):
    """Check if a given path matches any .gitignore pattern."""
    return any(regex.match(path) for regex in regex_patterns)

def scan_directory(base_path, gitignore_path):
    """Scan a directory and filter files/folders based on .gitignore rules."""
    regex_patterns = load_gitignore(gitignore_path)
    matched_files = []
    
    for root, dirs, files in os.walk(base_path):
        rel_root = os.path.relpath(root, base_path)
        if rel_root == '.':
            rel_root = ''
        
        if is_ignored(rel_root + '/', regex_patterns):
            continue  # Ignore entire directory
        
        for file in files:
            rel_path = os.path.join(rel_root, file)
            if is_ignored(rel_path, regex_patterns):
                continue
            matched_files.append(os.path.join(root, file))
    
    return matched_files

def build_tree(base_path, gitignore_path):
    """Build a dictionary tree of the directory structure excluding ignored files."""
    regex_patterns = load_gitignore(gitignore_path)
    tree = {}
    
    for root, dirs, files in os.walk(base_path, topdown=True):
        rel_root = os.path.relpath(root, base_path)
        if rel_root == '.':
            rel_root = ''
        
        if is_ignored(rel_root + '/', regex_patterns):
            dirs[:] = []  # Prevent os.walk from going deeper
            continue
        
        node = tree
        for part in rel_root.split(os.sep):
            if part and not is_ignored(part+'/', regex_patterns):
                node = node.setdefault(part, {})
        
        node.update({file: None for file in files if not is_ignored(os.path.join(rel_root, file), regex_patterns)})
    
    return tree

if __name__ == "__main__":
    base_directory = os.getcwd()  # Change this if needed
    gitignore_file = os.path.join(base_directory, ".gitignore")
    
    if not os.path.exists(gitignore_file):
        print(".gitignore file not found!")
    else:
        filtered_tree = build_tree(base_directory, gitignore_file)
        print(filtered_tree)
