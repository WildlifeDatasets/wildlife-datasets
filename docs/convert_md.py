import os

def convert_file(filename1, filename2):
    # TODO: name must be last
    # TODO: must be name="..."

    with open(filename1) as f:
        content = f.readlines()

    in_block = False
    setup = False
    blocks = {}
    content_new = []
    for line in content:
        print_line = True
        setup_finish = False    
        if not in_block and line.startswith('```'):
            line_split = line.split(' ')
            if line_split[-1].startswith('name'):
                line = ' '.join(line_split[:-1])
                line += '\n'
                content_new.append(line)
                print_line = False
                name = line_split[-1][6:-2]
                if name in blocks:
                    for qwe in blocks[name]:
                        content_new.append(qwe[:-1] + ' # markdown-exec: hide\n')
                else:
                    blocks[name] = []
            elif line_split[-1].startswith('setup'):
                setup = True
                name = line_split[-1][7:-2]
                if name in blocks:
                    raise(Exception('Setup already run'))
                else:
                    blocks[name] = []
            else:
                name = None
            in_block = True
        elif in_block and not line.startswith('```'):
            if name is not None and not line.endswith(' # markdown-exec: hide\n'): 
                blocks[name].append(line)
        elif in_block and line.startswith('```'):
            in_block = False
            setup_finish = True
        if print_line and not setup:
            content_new.append(line)
        if setup_finish:
            setup = False

    with open(filename2, 'w') as f:
        f.writelines(content_new)


filenames = [
    ["docs/.adding.md", "docs/adding.md"],
    ["docs/.tutorial_datasets.md", "docs/tutorial_datasets.md"],
    ["docs/.tutorial_splits.md", "docs/tutorial_splits.md"]
]

for filename1, filename2 in filenames:
    convert_file(filename1, filename2)
