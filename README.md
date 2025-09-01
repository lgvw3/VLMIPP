# VLM - MA - IPP

These are some early experiments into the concept of using VLMs for MA orchestration in the use case of IPP. 

The concept being explored is whether or not VLMs can effectively take a 2d map and agent positions, some basic belief map data, and orchestrate the agents in a unique or useful way

With this experimentation we hope to form the basis of experiments we can simulate in more robust simulators like Airsim. From there we would of course like to experiment in real situations.

## August 30, 2025

It seems, just from observing the decisions and entropy reduction that we need to improve the image information density for the VLM. I also think that persisting state, such as using openAIs conversations may improve quality or should at least be explored. Additionally the observations I am doing are done with 4.1-mini for cost purposes and perhaps a more powerful model will be needed but we need output to be robust and swift for real world.

## September 1st, 2025

Well probably won't have this read me go into a paper somewhere lol, I realized that the belief mean isn't really the objective function it is the reduction of the belief variance, which is the uncertainity, with is the information gain. After feeding in the belief variance and then tweaking it so that the variance map persists and isn't calculated entirely from scratch we are starting to see some useful skills emerge from asking a VLM to orchestrate.