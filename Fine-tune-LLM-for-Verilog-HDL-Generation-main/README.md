# Fine-tune-LLM-for-Verilog-HDL-Generation
## Overview 
The process of writing the synthesizable Verilog code for complex digital circuits from scratch is wearisome and error prone. To bridge the gap between high level design ideas and the final correct HDL code, we want to build a smart system that can take a simple, natural language description and generate the corresponding Verilog module.   
The project is motivated by the growing convergence of Artificial Intelligence and Electronic Design Automation (EDA). Capitalizing on the potential of large language models (LLMs), we aim to demonstrate connecting powerful AI models with fundamental digital design principles. This tool will act as an intelligent Design assistant, automating the most tedious part of the design flow-writing RTL code.  
## Techniques  
![my image](workflow_diagram.png)
This project was designed to be feasible with the constraints of a single GPU setup (specifically, a Nvidia TI 3080) therefore a 1 bit llama3 pre-trained model was used.    
*Dataset Curation:* The PyraNet-Verilog dataset was used as it had both prompts and output code. The dataset was scrubbed to include only code compiled successfully into ~150,000 samples. The stage 1 Fintuning used 40,000 samples fractioned by 4 categories of basic gates (NOT, AND, OR, NOR, XOR); adders, full adders ; MUXs ; and decoders, encoders so that a total of 50,000 final samples were used to finetune the data. 

*Fine-Tune with QLoRa (Quantization Low-Rank Adaptation):* QLoRA method of finetuning was used as a method to save memory with the GPU constraints while preserving accuracy. An initial stage 1 fine tuning was done on the 50,000 curated dataset. After initial testing, though the basic gates, adders, and MUXs give correct outputs, the decoders and encoders lacked accuracy. Therefore, an additional stage 2 fine tuning with a dataset of 10,000 decoder encoder samples was used to train the model, increasing the accuracy of the encoder decoders. 

*Optimization DPO (Direct Parameter Optimization):* To increase the precision of the outputs without having to train a reward model, a direct parameter optimization method was used. A randomized 10,000 samples were taken from the dataset, used as the prompt and chosen output. The prompt was then inputted in the fine-tuned model, checked that both outputs were not the same, then generated the rejected response. The fine-tuned model was then trained on the DPO dataset to increase the probability of the trained correct code as the output. 

*Verilog/HDL Generation:* A UI was created with the prompt, output, and waveform generation. The output was used as a testbench, (with simulation times, input patterns, outputs etc.) for Icarus Verilog to run the simulation, and MATLAB to plot and display the waveform on the UI. 

## Results 
The model was evaluated based on   
*Syntactic correctness:*  includes "module", "endmodule", inputs, outputs, compiles successfully (pass@5=0.8)  
*Functional Correctness:* correct output from prompt (pass@5=0.8)  
*Code Quality:* Has consistency in format (starts and ends with module and endmodule)  
Readable and concise (does not include extra comments or repeats)
## Conclusion
Overall, the model was successful in generating simple logic gates, adders, MUXs, and decoder/encoders. However it struggles with generating more complex circuits (adder with a MUX, higher order combinational logic design). Though we have a dynamic test bench to verify functionality, the waveforms are dependent on the correct output. Therefore in the future, a verification system should be in place either by a built in truth table or SystemVerilog Assertions. The GPU should also be increased so a 8 bit pre-trained model can be used in addition to a larger variety of complex circuit design data to achieve a better accuracy on higher order combinational circuit design. 
