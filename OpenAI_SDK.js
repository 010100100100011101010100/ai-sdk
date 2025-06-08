import {createOpenAI} from "@ai-sdk/openai";
import {generateText} from 'ai';
import {openai , OpenAIResponseProviderOptions} from "ai-sdk/openai";
import {fs} from 'fs';

const openai=createOpenAI({
    name:"Open Router",
    baseURL:"https://openrouter.com/api/v1",
    apiKey:"sdk-6717hjushgd",
    compatibility:"strict",
    headers:{
        'x-trace-id':'user-56677-session-9008',
        'x-custom-auth':'token'
    },
    organisation:"DcodeBlock",
    project:"project_Broski",
    fetch(url,init) => {
        console.log(`FETCH ${url}`);
        return fetch(url,init);
    };

})
//automatic detection of chat or completion API
const model=openai("gpt-4-turbo",{
    temperature:0.7,
    maxTokens:1000,
    system:"This is a question which you have to answer in easy manner",
});

const chatModel=openai.chat("gpt-3"{
    temperature:0.6,
    maxTokens:2000
})

const complModel=openai.completion("text-davinci-003",{
    temperature:1
})

const anotherExample=openai.chat("gpt-4",{
    temperature:0.9,
    maxTokens:3000,
    system:"You are a beautiful agent called Radhika",
    logitBias:{
        '19092':100,
        '9099':-100
    },
    user:"anotherUser",
});
//logprobs shows the confidence level of the model while word selection
//logprobs = True shows the word selected alongside with the log probability
//logprobs = number shows the word selected and the next number-1 it was going to select but did not
{
  "token": "Hello",
  "top_logprobs": {
    "Hello": -0.01,
    "Hi": -1.2,
    "Hey": -1.8,
    "Greetings": -2.1,
    "What's up": -2.4
  }
}
//These are log base e values (More closer to 0, more confidence, more -ve, it is unsure)
Math.exp(logprobs) -> Convert logprob values into probabilities


//parallelToolCalls -> Allow multiple to be called , safe if stateless (get APIs), otherwise DS manipulation takes time and less predictability
//legacyFunctionCalls is similar but restricts parallel calls and also is preferred with proxy AI setups

//structuredOutputs:True (Clean data in sync with schema DB config) -> better when generating JSON objects

//downloadImages: Boolean -> if the prompt contains image which are not shared, it will download it 



//some proxy models do not support token by token streaming. They send the message after it is fully generated
{simulateStreaming:true}


//reasoningEffort -> 3 types -> low,medium,high

//providerOptions -> global config seeting which you can set for all the models which you create with the AI provider
const {text,usage,providerMetadat}=await generateText({
    model:openai('gpt-3'),
    providerOptions:{
        openai:{
            reasoningEffort:"medium"
        },
    },
    prompt:"Tell me how bad am I stuck in this place"
})

console.log(text);
console.log('Usage',{
    ...usage,
    reasoningTokens:providerMetadata?.openai?.reasoningTokens //?. is optional and allows the code to not break if a value is NULL
})



//Structured Outputs for a prompt based on a schema according to which we want
import {z} from 'zod';
import {generateObject} from 'ai';
const result= await generateObject({
    model:openai('gpt-4',{
        structuredOutputs:true,
    }),
    schemaName:"recipe",
    schemaDescription:"This is the schema of how the recipe of lasgna is stored",
    schema:z.object({
        name:z.string(),
        ingredients:z.array(
            z.object({
                ingredientName:z.string(),
                amount:z.int()
            }),
        ),
        country:z.string(),
    }),
    prompt:"Generate a recipe for lasgna"

});

console.log(JSON.stringify(result.object,null,2));

//PDF SUPPORT FROM OPENAI

const PDFresult=await generateText({
    model:openai('gpt-4',{
        temperature:0.8,
        maxTokens:2000,
    }),
    messages:[
        {
            role:"user",
            content:[{
                type:"text",
                text:"Explain this entire pdf in easy manner"
            },
            {
                type:"file",
                mimeType:"application/pdf",
                data:fs.readFilySync('./data/ai.pdf'),
                name:'ai.pdf'
            },
        ],
        },
    ],
});
console.log(PDFresult)

//PREDICTIVE OUTPUTS in OPENAI Which allow user to generate faster responses, provide a better heuristic for the model to curate an output
const predictiveOp=generateText({
    model:openai('gpt-4'),
    prompt:"This is a predictive output prompt which I am typing",
    providerOptions:{
        openai:{
            prediction:"This is indeed a predictive prompt output"
        },
    }
})

console.log("Here in we use the prediction keyword ",predictiveOp);


//generateText function also returns model performance data in metadata
console.log(predictiveOp.providerMetadata.openai.acceptedPredictionTokens);
console.log(predictiveOp.providerMetadata.openai.rejectedPredictionTokens); // Detect when base prediction is off



//IMAGE DETAIL OPTION tells how the AI analyses an image- quickly (low) , high resolution (high), auto (based on the context)
const resultImage=await generateText({
    model:openai("gpt-4"),
    messages:[
        {
            role:'user',
            content:[
                {type:'text', text:"This is an image which I want you to understand"},
                {type:'image',image:'imageURL',
                providerOptions:{
                    openai:{imageDetail:'high'}
                }},
            ]
        },
    ]
});
console.log(resultImage);
//For UIchat option we use convertToCoreMessages function from ai sdk which translates all the content from the UI chat interface to the backend AI engine


//DISTILLATION : Larger model outputs answers and smaller,less resource consuming model trains on those answers (knowledge transfer). Feed same inputs, reduce output gap (distilation loss function) btw student and teacher model
//We can do so using logit matching, soft matching (confidence scores), temperature scaling, Intermediate layer matching (output + process)

async function main(){
    const {text,usage}=await generateText({
        model:openai('gpt-4o-mini'),
        prompt:'Who led DcodeBlock as a CEO ',
        providerOptions:{
            openai:{
                store:true, // stores the output which will be used for distillation
                metadata:{
                    custom:'value'
                }
            }
        }
    })
    console.log(text);
    console.log(usage)
}
main().catch(console.error);


//PROMPT CACHING - minimum 1024 and greater than that are cached, to avoid repititive computation and save resources and time
//providerMetadata allows us to see how many tokens in the prompt were taken from cached prompts, {cachedPromptTokens}
//Caching is done automatically BTW 
const {textFromcache,usageCache,metaData}=await generateText({
    model:openai("gpt-4o-mini"),
    prompt:"What do you think about the CEO of DcodeBlock"
});
console.log("Usage",{
    ...usageCache,
    cachedPromptTokens:providerMetadata?.openai?.cachedPromptTokens,
})


//AUDIO as an Input to the model - possible through openai-4o-audio-preview
const audioResult=await generateText({
    model:openai('gpt-4o-audio-preview'),
    messages:[
        {
            role:'user',
            content:[
                {type:'text',text:"What is this audio trying to mention"},
                {type:'file',
                mimeType:'audio/mpeg',
                data:fs.readFilySync('audio.mpeg')    ,
            }
            ]
        }
    ]
});
console.log(audioResult);



//Using Response method of OpenAI SDK which is better at handling multi modal input as well as agentic tool calls


//initialise the model
const gpt=openai.responses('gpt-4o=mini');
const responseResult=await model.invoke([
    {
    role:"user",
    content:[
        {
            type:'text',
            text:"This is an image which I want to understand"
        },
        {
            type:'image',
            image:"imageURL",
            mimeType:'image/png'
        }
    ]
    },
    {
        role:"assistant",
        toolCalls:[
            {
                name:'getWeather',
                args:{location:"New York"}
            }
        ]
    }
])


//Some PROVIDER OPTIONS :
`
1) parallelToolCalls - Multiple agents required
2) metadata
3) store - training
4) userId - rate limiting and other purposes
5) reasoningSummary - auto | detailed
6) reasoningEffort - low | medium |high
7) strictSchemas - particular JSON input and output format
8) previousResponseId - Manage Context History of the chat  

`

//WEB SEARCH - uses openai responses provider and webSearchPreview tool
const webResults=await generateText({
    model:openai.responses('gpt-4o-mini'),
    prompt:"What is the time right now in Ukraine",
    tools:{
        webSearchPreview:openai.tools.webSearchPreview({
            searchContextSize:'High'
        })
    },
    toolChoice:{ type:'tool',toolName:"webSearchPreview"}
});

console.log(webResults.sources);




































