from gan import GAN

def main(EPOCHS=500, BATCH_SIZE=100, GPU_ID=1, SEED=2022):
    model = GAN(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, GPU_ID=GPU_ID, SEED=SEED)
    model.train()
    
if __name__ == "__main__":
    main()