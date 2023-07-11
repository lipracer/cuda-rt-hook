import sys, re, os

g_lib_name = "/opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so"
def addr2line(addr):
    s = f"addr2line -e {g_lib_name} -f {addr}"
    with os.popen(s) as pf:
        symbol = pf.readlines()
        return [s.strip() for s in symbol]

def demangler(symbol):
    end = symbol.find("+0x")
    if end != -1:
        symbol = symbol[:end]
    s = f"c++filt {symbol}"
    with os.popen(s) as pf:
        symbol = pf.readlines()[0]
        return symbol.strip()

def parse_and_filter(file):
    lines = []
    with open(file, "rt") as f:
        for line in f.readlines():
            base_addr_int = 0
            if line.find("ignore:[base address]:") != -1:
                base_addr = line[len("ignore:[base address]:"):]
                base_addr_int = int(base_addr, 16)
            if line.find("ignore") != -1:
                lines.append(line)
                continue
            if line.find("libtorch_cuda.so") == -1:
                continue
            def repl(g):
                file_addr = hex(int(g.group(3), 16) - base_addr_int)
                # info = addr2line(g.group(1)[1:])[0]
                info = addr2line(file_addr)
                symbol = info[0]
                sorce_file = info[1]
                result = g.group(0).replace(g.group(2), symbol)
                return result.replace(g.group(1), sorce_file)
            line = re.sub(r"(.*libtorch_cuda.so)\((\+.*?)\) \[(.*?)\]", repl, line)
            lines.append(line)
    with open(file, "wt") as f:
        for line in lines:
            f.write(line)

def main():
    # g_lib_name = sys.argv[1]
    # file = sys.argv[2]
    file = sys.argv[1]
    parse_and_filter(file)
    # lines = []
    # with open(file, "rt") as f:
    #     for line in f.readlines():
    #         def repl(g):
    #             return g.group(0).replace(g.group(1), demangler(g.group(1)))
    #         line = re.sub(r".*libtorch_cuda.so\(([\s|\S]*?)[\)|\n]", repl, line, re.M)
    #         lines.append(line)
    # with open(file, "wt") as f:
    #     for line in lines:
    #         f.write(line)



if __name__ == "__main__":
    main()
