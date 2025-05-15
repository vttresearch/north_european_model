import subprocess
import sys
import os


def main():

    for input_path in input_paths:
        output_path = f'./toolbox_workflow/{os.path.basename(input_path)[:-5]}.gdx'
        cmd_call = f'gdxxrw Input={input_path} Output={output_path} Index = Index!'
        os.system(cmd_call)

if __name__ == "__main__":
    input_paths = sys.argv[1:]
    main()