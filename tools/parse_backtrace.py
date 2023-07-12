import sys, re, os

# g_lib_name = "/opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so"
def addr2line(addr):
    s = f"addr2line -e {g_lib_name} -f {addr}"
    with os.popen(s) as pf:
        symbol = pf.readlines()
        return [s.strip() for s in symbol]

def demangler(symbol):
    if len(symbol) == 0:
        return ''
    end = symbol.find("+0x")
    if end != -1:
        symbol = symbol[:end]
    else:
        return ''
    s = f"c++filt {symbol}"
    with os.popen(s) as pf:
        symbol = pf.readlines()[0]
        return symbol.strip()

def parse_and_filter(file):
    new_lines = []
    with open(file, "rt") as f:
        lines = f.readlines()
        first_line = lines[0]

        base_addr = first_line[len("ignore:[base address]:"):]
        base_addr_int = int(base_addr, 16)
        for line in lines[1:]:
            if line.find("ignore") != -1:
                lines.append(line)
                continue
            if line.find("libtorch_cuda.so") == -1:
                lines.append(line)
                continue
            def repl(g):
                file_addr = hex(int(g.group(3), 16) - base_addr_int)
                # info = addr2line(g.group(1)[1:])[0]
                info = addr2line(file_addr)
                symbol = info[0]
                sorce_file = info[1]
                result = g.group(0).replace(g.group(1), sorce_file)
                if g.group(2)[0] == "+":
                    result = result.replace(g.group(2), symbol)
                return result
            line = re.sub(r"(.*libtorch_cuda.so)\((.*)\) \[(.*?)\]", repl, line)
            new_lines.append(line)
    with open(file, "wt") as f:
        for line in new_lines:
            f.write(line)

def main():
    g_lib_name = sys.argv[1]
    file = sys.argv[2]
    file = sys.argv[1]
    parse_and_filter(file)
    
    with open(file, "rt") as f:
        lines = []
        for line in f.readlines():
            if line.find("ignore") != -1:
                lines.append(line)
                continue
            def repl(g):
                return g.group(0).replace(g.group(1), demangler(g.group(1)))
            line = re.sub(r".*?\(([\s|\S]*?)[\)|\n]", repl, line, re.M)
            lines.append(line)
    with open(file, "wt") as f:
        for line in lines:
            f.write(line)



if __name__ == "__main__":
    main()
