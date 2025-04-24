import torch


def log_parameters(func):
    def wrapper(*args, **kwargs):
        print(f"Start call: {func.__name__}")

        def gen_tenor(t: torch.Tensor):
            return f"torch.randn({tuple(t.shape)}, dtype={t.dtype})"

        # args
        gen_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                gen_args.append(gen_tenor(arg))
            elif isinstance(arg, int):
                gen_args.append(str(arg))
            else:
                gen_args.append(arg)

        # kwargs
        gen_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                gen_kwargs[key] = gen_tenor(value)
            elif isinstance(value, int):
                gen_kwargs[key] = str(value)
            else:
                gen_kwargs[key] = value

        code = "arg: \n"
        code += f"{gen_args}"
        code += "\nkwargs: \n"
        code += f"{gen_kwargs}"
        print(code)
        print(f"End call: {func.__name__}")
        # return func(*args, **kwargs)

    return wrapper
