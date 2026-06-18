import torch
import numpy as np
'''
Profiling and interpreting profiler is important to learn early
Goals:
-Understand some basics of profiling CUDA kernels in Torch programs
-Understand how to write Cuda C/C++ in Torch
-Acquire basic understanding of NCU and modern profiling tooling for ML workflows

'''

#Rough implementation of autograd profiler, just see elapsed time for individual funcs
def profiler(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        func(input) #Input multiple time events on our function and input

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)

#Pytorch Implementation (obviously abstracted by function calls, just useful to see without schedules and warmpus)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        a = torch.square(torch.randn(10000, 10000).cuda())
prof.export_chrome_trace("trace.json")

'''Flow events: Shows the direction of the program execution aten:square --> aten:pow --> cuda kernel,
                PyTorch Does not quite show how fast the kernels are or what they do, helps us know the name and the flows'''

''' PyBind: Create Python bindings for C++,

load_inline: Cleaner version of achieving PyBinds, codegens everything for you'''

cpp_source = """
std::string foo() {
    return "Is this the Pythonic World?"
    }
"""

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['foo'],
    verbose=True

)

print(my_module.foo())





