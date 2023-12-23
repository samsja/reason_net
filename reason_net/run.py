from reason_net.data import MathDataGen, MathTokenizer


def run():
    generator = MathDataGen(0, 3)
    tokenizer = MathTokenizer(MathDataGen.operand)

    for _ in range(10):
        data, target = generator.generate()
        print(data, target)

        encode = tokenizer.encode(data)
        print(encode)
        print(tokenizer.decode(encode))


if __name__ == "__main__":
    run()
