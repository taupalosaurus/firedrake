from pyadjoint.overloaded_type import FloatingType
from .blocks import DirichletBCBlock
from pyadjoint.tape import no_annotations


class DirichletBCMixin(FloatingType):
    @staticmethod
    def _ad_annotate_init(init):
        def wrapper(self, *args, **kwargs):
            FloatingType.__init__(self,
                                  *args,
                                  block_class=DirichletBCBlock,
                                  _ad_args=args,
                                  _ad_floating_active=True,
                                  **kwargs)
            init(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def _ad_annotate_apply(apply):
        @no_annotations
        def wrapper(self, *args, **kwargs):
            for arg in args:
                if not hasattr(arg, "bcs"):
                    arg.bcs = []
            arg.bcs.append(self)
            return apply(self, *args, **kwargs)
        return wrapper

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        if len(deps) <= 0:
            # We don't have any dependencies so the supplied value was not an OverloadedType.
            # Most probably it was just a float that is immutable so will never change.
            return None

        return deps[0]

    def _ad_restore_at_checkpoint(self, checkpoint):
        if checkpoint is not None:
            self.set_value(checkpoint.saved_output)
        return self
