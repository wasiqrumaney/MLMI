# VAE

## vert32

### TODO
* ~~save/load checkpoint~~
* **beta VAE**   tried, but correct?
* improve vae32 architecture
* better train function
* **latent space navigation**
* **try Dice loss**
* try perceptual loss (pre-trained 3d net?)
* **infoVAE** tried, but not working
#### Visualization
* FOR MESH PLOT: colormap based on probs
* ~~matplotlib3d voxels visualization~~ better transparency? 
* **save results snapshots**



## vert64
Next step after 32.

### TODO
* fix architecture bottleneck


## Notes on Tensorboard Visualization
In order not to use it set USE_TBX to False
To use it, install tensorflow, tensorboard and tensorboardX and set USE_TBX to True. <br/>
To show the graphs run in the directory with the runs folder<br/>
`tensorboard --logdir=./logs/<TRIAL_ID> --host localhost --port 8088`<br/>
and go to localhost:8088 in browser.