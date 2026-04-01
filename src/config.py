class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    z_dim = 100            # noise dimension
    y_dim = 10             # number of classes (such as MNIST)
    g_hidden_dim = 128
    
    d_hidden_dim = 128
    
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 50
