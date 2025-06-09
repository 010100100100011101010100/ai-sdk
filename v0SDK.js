//Start of v0 SDK using vercel ai-sdk


//v0 has the v0-1.0-md model which I love man (text and image as inputs) . Allows streaming responses

import {createVercel} from "@ai-sdk/vercel";
import {vercel,generateText} from 'ai';

const model=vercel('v0-1.0-md');



//Custom config of the vercel client
const client=await createVercel({
    apiKey:process.env.VERCEL_API_KEY??' ',
    baseURL:"selfHostedServer.com",
    header:{
        'Authorisation':apiKey,
    },
    fetch:async(input,init)=>{
        console.log("This is the request my Nigga",input);
        fetch(input,init);
    }
});



//generateText Function
const result=await generateText({
    model,
    prompt:"Create a Next JS app for me which says GM",
});
console.log(result);

//Vercel has Autofix,Figma reading and Image analysis capabilities. The Newer model has ability to generate placeholder images as well.


