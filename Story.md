
I set off with the goal of making an LLM that sounds like me. 

I first started by making an export of all my imessages into one big txt document. This was every message that I have sent. 

Then on this, I ran CPT (continued pre-training) which ended up with the model (Qwen 4B) giving similar sounding responses, but it didn't feel quite right. 

Often it would hallucinate and say randomness.

Here's an example of some back & forth with the ai.

Then I tried SFT (supervised-fine-tuning) by making the data in the format of previous-message and response.

Here's what the result looked like:

I was doing all the training and running of the model locally on my laptop to this point. I mainly did this out of security concern--I don't want my messages hosted on someone else's server. 

I thought maybe I did not have enough data. So I went to an online community I've been part of for a few years and exported all the data of everything I ever said there. It was 10s of thousands of messages. This big text blob was then used for more CPT on QWEN. 

Here's what the result was like:

This time I used tinker because I didn't mind public messages going onto their training server. So the resulting model was a bit sloppy sometimes but it definitely sounded like how I chat on that server. 

Then I tried making it do RL next-token prediction instead of CPT and tbh the results were pretty bland because it almost never got the next token right so the rewards were too scarce for the model to see significant change. 

I realized I want to do PEFT locally and do the SFT on the Tinker server so my messages never go onto their server & so I tried this and instead of just doing the previous message and the next message, I did the context of the previous 5 messages. 

Ok now with 15 message context, formatting for message format, and SFT, it's doing pretty well!

I now am going to play with my two levers: learning rate and lora

I am going to automate this part where I basically let it generate several models with learning rate & lora modules, have back-and-forth eval where I have them walk through various scenarios (a friend inviting you to go on a hike, a friend confiding in you and asking you for advice, a person reaching out after an event scheduling a time to go to a cafe, a friend reaching out asking when we can do the podcast).

Rate conversations on how-realistic, how-kind, how-casual, how-concise, how-repetitive, how natural, etc.

And it should save all of it based on these different learning rate & lora combos so it is clear. 

Make so it's interacting with a little gemini model back and forth in these convos and don't make them more than 10-messages max. Make so gemini can end it at end point. 

- LR: `1e-4`, `1.5e-4`, `2e-4` — all in a tight, common LoRA-ish band (only 2× from lowest to highest). That’s not “near zero vs enormous”; it’s “conservative vs slightly more aggressive.”
- Rank: `8`, `16`, `32` — small → medium adapters (powers of two, Tinker-friendly). You’re not testing rank `4` vs `128`; you’re testing meaningful steps without the cost/risk of huge ranks.

I realize I am actually doing a small band instead of a large band. I want to do larger testing range. 

lr0p0002_r8 — learning rate 2e-4, LoRA rank 8, base model openai/gpt-oss-120b (Tinker LoRA checkpoint imessage-sft-lora-lr0p0002-r8), mean rubric ≈ 3.75.

Ok I am going to go to the grocery store, but I am making an agent loop so that while I am gone, Claude has freedom to experiment with a larger band and narrow in on parts that seem interesting. I am making the training a bit more consistent because I think part of the variability from before may have been the model changing things up. 

Ok here were the results and the best model that I found...