//Start of xAI sdk 
//npm install @ai-sdk/xai


import {xai} from '@ai-sdk/xai'; //provider instance option
import {createxAI} from '@ai-sdk/xai' //Instance Creator Helper function for a customised setup
import {generateText} from "ai";
import {experimental_image as generateImage} from 'ai';



const xai=createxAI({
    apiKey:"randomassAPIkey",
    baseURL:"proxyserver.com", //self hosted or proxy server
    headers:{
        'X-custom-headers':'value'
    },
    fetch: async(input,init)=>{ //logging,caching and testing purposes
        console.log('Outgoing reqest',input);
        return fetch(input,init);
    }
});

//language Model -> groq-3
const xai1=await generateText({
    model:xai("groq-3"),
    prompt:"What is the best song of the century"
});

//chat model -> personal chatting with a specified user

//specify model along with user config
const model=xai('groq-3',{
    user:"raseshGautam",
});


const chatxAI=await generateText({
    model,
    messages:[
        {role:'user',content:'What do you think about love?'}
    ],
    providerOptions:{
        xai:{
            reasoningEffort:'high'
        },
    },
});
console.log(chatxAI.result);



//Image gen capabilities with groq-vision and groq-2-image
//Standard dimensions and apparent depth = 1024*768


const imageResult=await generateImage({
    model:xai.image("groq-2-image",{
        maxImagesPerCall:5
    }),
    prompt:"Create the image of the most beatiful couple on a beachside",
    n:2,//this is how many versions of the image I would need
})

console.log(imageResult.result);












