from autorag_live.core.di import DIRegistry


def test_di_registry_resolve():
    reg = DIRegistry()

    class A:
        def __init__(self):
            self.x = 1

    reg.register("a", lambda: A())
    inst = reg.resolve("a")
    assert isinstance(inst, A)

    reg.register("single", lambda: A(), singleton=True)
    s1 = reg.resolve("single")
    s2 = reg.resolve("single")
    assert s1 is s2
