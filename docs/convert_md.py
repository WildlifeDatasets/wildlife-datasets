import numpy as np

def convert_file(filename1: str, filename2: str) -> None:
    """Converts files allowing to use named blocks for markdown-exec.

    The named blocks must be in the form of
        ```python some_attribute="" name="block_name" other_attribute=""
    The script merges the blocks and hides the lines_old of the previous blocks.    
    This is highly inefficient as it runs the same code multiple times.
    The name of the original file should start with `.` (it is ignored when building documentation).

    Args:
        filename1 (str): Name of the original file with named blocks.        
        filename2 (str): Name of the converted file suitable for markdown-exec.
    
    Examples:
        Input file:
            ```python exec="true" source="above" result="console" name="name1"
            x = 1
            print(x) # markdown-exec: hide
            ```

            ```python exec="true" source="above" result="console" name="name1"
            y = x + 1
            print(y) # markdown-exec: hide
            ```

        Output file:
            ```python exec="true" source="above" result="console" name="name1"
            x = 1
            print(x) # markdown-exec: hide
            ```

            ```python exec="true" source="above" result="console" name="name1"
            x = 1 # markdown-exec: hide
            y = x + 1
            print(y) # markdown-exec: hide
            ```
    """

    # Load the original file
    with open(filename1) as f:
        lines_old = f.readlines()

    in_block = False # Checks whether we are in a block started by ```
    blocks = {}  # Dictionary of the saved blocks. Keys are block names
    lines_new = [] # Lines of the new file
    for line in lines_old:
        if not in_block and line.startswith('```'):
            # Starting a new block (header)
            in_block = True
            line_split = line.split(' ')
            ii = np.where([x.startswith('name') for x in line_split])[0]
            # Checks whether it is a named block
            if len(ii) == 1:
                # Extract the name of the block and remove it from the header
                block_name = line_split.pop(ii[0])
                block_name = block_name[6:]
                line = ' '.join(line_split)
                if block_name.endswith('\n'):
                    line += '\n'
                    block_name = block_name[:-2]
                else:
                    block_name = block_name[:-1]
                # Add the content of the previous blocks with the same name to the new file
                lines_new.append(line)
                if block_name in blocks:
                    for block_line in blocks[block_name]:
                        lines_new.append(block_line[:-1] + ' # markdown-exec: hide\n')
                else:
                    blocks[block_name] = []
            else:
                block_name = None
                lines_new.append(line)
        elif in_block and not line.startswith('```'):
            # Inside a block
            if block_name is not None and not line.startswith('print'): 
                blocks[block_name].append(line)
            lines_new.append(line)
        elif in_block and line.startswith('```'):
            # Finishing a block (footer)
            in_block = False
            lines_new.append(line)
        else:
            # Outside of a block
            lines_new.append(line)

    # Save to the converted file
    with open(filename2, 'w') as f:
        f.writelines(lines_new)


filenames = [
    ["docs/.adding.md", "docs/adding.md"],
    ["docs/.default_splits.md", "docs/default_splits.md"],
    ["docs/.methods.md", "docs/methods.md"],
    ["docs/.recommended.md", "docs/recommended.md"],
    ["docs/.tutorial_datasets.md", "docs/tutorial_datasets.md"],
    ["docs/.tutorial_splits.md", "docs/tutorial_splits.md"],
]

for filename1, filename2 in filenames:
    convert_file(filename1, filename2)

