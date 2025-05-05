/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { type FunctionDeclaration, SchemaType } from "@google/generative-ai";
import { useEffect, useRef, useState, memo } from "react";
import vegaEmbed from "vega-embed";
import { useLiveAPIContext } from "../../contexts/LiveAPIContext";
import { ToolCall } from "../../multimodal-live-types";

const altairDeclaration: FunctionDeclaration = {
  name: "render_altair",
  description: "Displays an altair graph in json format.",
  parameters: {
    type: SchemaType.OBJECT,
    properties: {
      json_graph: {
        type: SchemaType.STRING,
        description:
          "JSON STRING representation of the graph to render. Must be a string, not a json object",
      },
    },
    required: ["json_graph"],
  },
};

const checkWorkDeclaration: FunctionDeclaration = {
  name: "check_work",
  description: "Analyzes an image for mathematical or logical mistakes in written work.",
  parameters: {
    type: SchemaType.OBJECT,
    properties: {
      validation_results: {
        type: SchemaType.STRING,
        description: "JSON STRING representation of the validation results, containing an array of objects. Each object has 'x', 'y', 'width', 'height' (defining a bounding box for a step), and 'correct' (boolean indicating if the step is mathematically/logically correct)."
      },
    },
    required: ["validation_results"],
  },
};

function AltairComponent() {
  const [jsonString, setJSONString] = useState<string>("");
  const [validationResults, setValidationResults] = useState<string>("");
  const { client, setConfig } = useLiveAPIContext();

  useEffect(() => {
    setConfig({
      model: "models/gemini-2.0-flash-exp",
      generationConfig: {
        responseModalities: "audio",
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: "Aoede" } },
        },
      },
      systemInstruction: {
        parts: [
          {
            text: 'You are my helpful assistant. Any time I ask you for a graph call the "render_altair" function I have provided you. When I say "check my work" or similar phrases, analyze any attached images for mathematical or logical errors using the "check_work" function. Provide step-by-step feedback on any mistakes found.',
          },
        ],
      },
      tools: [
        // there is a free-tier quota for search
        { googleSearch: {} },
        { functionDeclarations: [altairDeclaration, checkWorkDeclaration] },
      ],
    });
  }, [setConfig]);

  useEffect(() => {
    const onToolCall = (toolCall: ToolCall) => {
      console.log(`got toolcall`, toolCall);
      
      // Handle Altair graph rendering
      const altairCall = toolCall.functionCalls.find(
        (fc) => fc.name === altairDeclaration.name,
      );
      if (altairCall) {
        const str = (altairCall.args as any).json_graph;
        setJSONString(str);
      }
      
      // Handle work validation
      const checkWorkCall = toolCall.functionCalls.find(
        (fc) => fc.name === checkWorkDeclaration.name,
      );
      if (checkWorkCall) {
        const results = (checkWorkCall.args as any).validation_results;
        setValidationResults(results);
        // You would likely want to render these validation results in the UI
        // This could be done by parsing the JSON and drawing boxes on a canvas overlay
        console.log("Validation results:", JSON.parse(results));
      }
      
      // Send response for all tool calls
      if (toolCall.functionCalls.length) {
        setTimeout(
          () =>
            client.sendToolResponse({
              functionResponses: toolCall.functionCalls.map((fc) => ({
                response: { output: { success: true } },
                id: fc.id,
              })),
            }),
          200,
        );
      }
    };
    client.on("toolcall", onToolCall);
    return () => {
      client.off("toolcall", onToolCall);
    };
  }, [client]);

  const embedRef = useRef<HTMLDivElement>(null);
  const validationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (embedRef.current && jsonString) {
      vegaEmbed(embedRef.current, JSON.parse(jsonString));
    }
  }, [embedRef, jsonString]);

  // Optional: Render validation results if you want to visualize them
  useEffect(() => {
    if (validationRef.current && validationResults) {
      // Here you could implement rendering of the validation results
      // For example, drawing boxes around incorrect steps in an image
    }
  }, [validationRef, validationResults]);

  return (
    <div>
      <div className="vega-embed" ref={embedRef} />
      <div className="validation-overlay" ref={validationRef} />
    </div>
  );
}

export const Altair = memo(AltairComponent);
