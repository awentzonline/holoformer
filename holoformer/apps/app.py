import argparse

try:
    from .utils import set_device_from_args
except ImportError:
    pass


class App:
    app_name = 'base app'

    def __init__(self, args):
        self.args = args

    def start(self):
        self.setup()
        self.run()

    def setup(self):
        pass

    def run(self):
        pass

    @classmethod
    def run_from_cli(cls):
        app = cls.create_from_cli()
        app.start()

    @classmethod
    def create_from_cli(cls):
        args = cls.parse_cli_args()
        obj = cls(args)
        return obj

    @classmethod
    def parse_cli_args(cls):
        p = argparse.ArgumentParser(
            description=cls.app_name, fromfile_prefix_chars='@'
        )
        cls.add_args_to_parser(p)
        return p.parse_args()

    @classmethod
    def add_args_to_parser(cls, p):
        pass


class AppWithDevice(App):
    """Saves you the trouble of updating args with the desired device."""
    def __init__(self, app_args, *args, **kwargs):
        super().__init__(app_args, *args, **kwargs)
        set_device_from_args(app_args)
        print(f'Using device "{app_args.device}"')

    @classmethod
    def add_args_to_parser(cls, p):
        super().add_args_to_parser(p)
        p.add_argument('--no-cuda', action='store_true')
