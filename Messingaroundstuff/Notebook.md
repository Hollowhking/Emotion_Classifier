final project:
something that records 30sec of audio then runs it through the ML network and returns the current emotional state

Things to do:
- read the research papers(also takes notes on how the research papers are structured)
- In the paper bring up how the model reacts to wild datasets, lab datasets, training with both
- The final train will be a dataset with only me

  **WRITING A PAPER:**
The goal of a literature review is to learn what’s currently being done in the field, how they’re doing it, and to then identify a gap in the literature (something that hasn’t been done). You may find it helpful to take notes in a central document. When you begin writing up your thesis document, you’ll need to include a written literature review, which will position (i.e. motivate) your current project.  

For example, X did 1, Y did 2, Z did 3, but no one’s done 4, so I’m going to do 4 for reasons a, b, and c.
# Topic:#
Speech emotion recognition specializing range of sad emotions. (the way you speak is similar to a facial expression -Dr livingstone)

# Final Project:#
Build a emotion classifier & a program that can record 30 seconds of speech run it through the classifier and determine the emotion of the speech
# What features does opensmile take:#
- Open smile's eGeMAPSv02 returns 88 features when classifiers are trained with the 88 set vs the 62 set made with
- Features that stay the same and ones that change and why given different data 
## Different classifiers to test:##
	Perceptron, Decision tree, Logistic regression, deep learning. 
## Finished Items:##
created a recorder that records for 5 seconds and saves to a .wav file

# Thesis statement:#
	I built a classifier that looks at the range of negative emotions and placing them on a spectrum for the goal of identifying the difference of states from voice alone. SPEECH PATTERNS
# Good vibrations:  Reference features in voice and anatomy of voice#

-this paper (on page 13) establishes acoustic patterns of positive emotions.
- The measurement of acoustic parameters in emotional vocal expressions focuses on 3 domains:
    -frequency
    -amplitude(intensity)
    -duration/speech rate
    (These features would be nearly identical to the ones used for the sadness vibrations)
-talks about the source of speech is the larynx and the pharynx/oral cavity serve as filters to the audio
-common parameters for acoustic studying,

![[Pasted image 20231011202059.png]]

Voice quality is the amount of vibration at each frequency. these measures are used to indicate voice stability a calm normal voice will likely have small amounts of instability.

Most studies came to the conclusion, someone feeling happiness will have a higher f0 (frequency), a higher f1 and f2 as well. However no variation in speech rate was determined.

Writing a research paper notes:
- referring to something as a variable as long as the variable name is explained and well understood

- figures: take a sad sample, plot its pitch on a graph, take a happy sample, plot its pitch and point out the differences.


# Speech emotion recognition#

* The research done in this paper is on the current research in the field of SER and the challenges faced in that research
* Details the 6 states of basic emotion: sadness, happiness, fear, anger, disgust, and surprise
For the sake of this research we will be focusing on sadness/fear as these emotional states are used to convey sad emotions. These emotional states are detailed more in: (Ekman et al. (2013) which details the facial expressions tied to those emotions. 

(Watson et al., 1988).) details the latent features of speech being: valence, arousal, control, and power

* The problems with the different types of databases and the lack of databases containing data on elderly and young people

* this paper also references the difference in emotional recognition with different languages, for example German as a language is commonly described as an angry language thus a SER system trained on English would likely classify all German recordings as angry. 

* Types of features analyzed: 
	* Prosodic: things that are actively noticed by people like intonation and rhythm of speech this is commonly used to determine if something is a question. Speech rate, duration of silence regions, rate of duration of voiced and unvoiced regions, duration of longest voiced speech
	* Spectral: the shape of the vocal cords and how that changes the sound 
	* Voice quality

# A comparative study of different classifiers for detecting depression from spontaneous speech#
Uses opensmile to extract several acoustic descriptors or LLD low-level descriptors. 

* 4 main classifiers: 
	* Gaussian mixture models: 
		* Generative classifier common for speech analysis. It models low level frame based features directly, regardless of duration. Trained using a continuous hidden markov model, with a single state using 16 weighted mixtures of gaussian densities. 
	* Support vector machines:
		* SVM is a discriminative classifier, it provides good generalization. however it doesn't function as well with clips that are different lengths.
	* Hierarchical fuzzy signature
	* Multilayer perceptron neural network


# Self-report captures 27 distinct categories of emotion bridged by continuous gradients:#
Proposes 27 catagories that bridge all emotions together and that all emotions are combinations of these 27. 
The important ones for me are::
	F- Anxiety (Fear, uncertainty)
	R - Horror
	X - Sadness
Even on this scale sadness is viewed as a monolith while i propose that this category is so much broader than the rest 

# A review of depression and suicide risk assessment using speech analysis:#





# ASPECTS OF THESIS WORK RIGHT NOW:#

### What can i do better ###

Timeline: 
	December -> Finish The emotion classifier
	January -> Tie in the binary depression classifier
	February -> Finish the classifier
	March -> Finish Thesis
	April -> Polish the thesis
## Building the classifier: ##
 Binary depression classifier FEATURE EXTRACTION. EXPLAIN FEATURES. 
## Writing the thesis:##
- Pre and post (soley train model on basic emotion model, then train it on depressed dataset)
## Finding more apt Datasets:##
Find datasets that contain depressed people. 