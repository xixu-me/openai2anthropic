// Anthropic to OpenAI Proxy Worker

// Constants will be replaced by environment variables
// Default values used if not set in wrangler.toml

// Main handler for incoming requests
export default {
	async fetch(request, env, ctx) {
		// Only allow POST requests
		if (request.method !== 'POST') {
			return new Response('Method not allowed', { status: 405 });
		}

		// Ensure the request is to the correct endpoint
		const url = new URL(request.url);
		if (!url.pathname.endsWith('/messages') && !url.pathname.endsWith('/completions')) {
			return new Response('Not found', { status: 404 });
		}

		// Validate API Key
		const authHeader = request.headers.get('Authorization');
		if (!authHeader || !authHeader.startsWith('Bearer ')) {
			return new Response(
				JSON.stringify({
					error: {
						type: 'authentication_error',
						message: 'Missing or invalid Authorization header. Please provide a valid API key.',
					},
				}),
				{
					status: 401,
					headers: { 'Content-Type': 'application/json' },
				}
			);
		}

		const apiKey = authHeader.slice(7); // Remove 'Bearer ' prefix

		// Validate against configured API keys (you can configure multiple keys)
		const validApiKeys = env.ANTHROPIC_API_KEYS ? env.ANTHROPIC_API_KEYS.split(',') : [];
		if (validApiKeys.length === 0 || !validApiKeys.includes(apiKey)) {
			return new Response(
				JSON.stringify({
					error: {
						type: 'authentication_error',
						message: 'Invalid API key provided.',
					},
				}),
				{
					status: 401,
					headers: { 'Content-Type': 'application/json' },
				}
			);
		}

		try {
			// Clone the request for reading
			const requestClone = request.clone();
			const anthropicRequest = await requestClone.json();

			// Check if this is a streaming request
			const stream = anthropicRequest.stream === true;

			// Convert Anthropic request to OpenAI format
			const openaiRequest = convertAnthropicToOpenAI(anthropicRequest, env);

			// Send the transformed request to OpenAI
			const openaiResponse = await fetch(env.OPENAI_API_URL, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${env.OPENAI_API_KEY}`,
				},
				body: JSON.stringify(openaiRequest),
			});

			// Handle errors from OpenAI
			if (!openaiResponse.ok) {
				const error = await openaiResponse.json();
				return new Response(
					JSON.stringify({
						error: {
							type: 'proxy_error',
							message: `OpenAI API error: ${error.error?.message || 'Unknown error'}`,
						},
					}),
					{
						status: openaiResponse.status,
						headers: { 'Content-Type': 'application/json' },
					}
				);
			}

			// Handle streaming responses
			if (stream) {
				// Create a TransformStream to convert OpenAI stream format to Anthropic stream format
				const { readable, writable } = new TransformStream();

				// Process the stream
				processStream(openaiResponse.body, writable);

				// Return the transformed stream
				return new Response(readable, {
					headers: {
						'Content-Type': 'text/event-stream',
						'Cache-Control': 'no-cache',
						Connection: 'keep-alive',
					},
				});
			} else {
				// For non-streaming, convert the full response
				const openaiData = await openaiResponse.json();
				const anthropicResponse = convertOpenAIToAnthropic(openaiData, anthropicRequest);

				return new Response(JSON.stringify(anthropicResponse), {
					headers: { 'Content-Type': 'application/json' },
				});
			}
		} catch (error) {
			// Handle any errors
			return new Response(
				JSON.stringify({
					error: {
						type: 'proxy_error',
						message: `Error processing request: ${error.message}`,
					},
				}),
				{
					status: 500,
					headers: { 'Content-Type': 'application/json' },
				}
			);
		}
	},
};

// Convert Anthropic request format to OpenAI format
function convertAnthropicToOpenAI(anthropicRequest, env) {
	// Use the model defined in environment variables instead of mapping
	// Ignore the model requested in the Anthropic API
	const openaiModel = env.OPENAI_MODEL_ID;

	// Build messages array
	const messages = [];

	// Handle both legacy completions API and messages API
	if (anthropicRequest.messages) {
		// Claude Messages API
		for (const message of anthropicRequest.messages) {
			if (message.role === 'assistant' && message.content.length === 0) {
				// Skip empty assistant messages (sometimes used as prompt terminators)
				continue;
			}

			// Map role (Anthropic uses 'user' and 'assistant', OpenAI also has 'system')
			let role = message.role;

			// Convert message content
			let content;
			if (typeof message.content === 'string') {
				content = message.content;
			} else if (Array.isArray(message.content)) {
				// Handle content parts (text and images)
				const parts = message.content;

				// If there's only text parts, simplify to string
				const onlyTextParts = parts.every((part) => part.type === 'text');
				if (onlyTextParts) {
					content = parts.map((part) => part.text).join('');
				} else {
					// Handle mixed content including images
					content = parts
						.map((part) => {
							if (part.type === 'text') {
								return { type: 'text', text: part.text };
							} else if (part.type === 'image') {
								return {
									type: 'image_url',
									image_url: {
										url: part.source.url || part.source.data,
										detail: part.source.media_type === 'image/png' ? 'high' : 'low',
									},
								};
							}
							return null;
						})
						.filter((item) => item !== null);
				}
			}

			messages.push({ role, content });
		}

		// Handle system prompt if present
		if (anthropicRequest.system) {
			messages.unshift({ role: 'system', content: anthropicRequest.system });
		}
	} else if (anthropicRequest.prompt) {
		// Legacy Claude Completions API
		// Extract user/assistant messages from the prompt string
		const parts = anthropicRequest.prompt.split(/\n\nHuman: |\n\nAssistant: /);
		let isHuman = anthropicRequest.prompt.trim().startsWith('Human:');

		for (let i = 0; i < parts.length; i++) {
			const part = parts[i].trim();
			if (part.length === 0) continue;

			const role = isHuman ? 'user' : 'assistant';
			messages.push({ role, content: part });
			isHuman = !isHuman;
		}
	}

	// Build OpenAI request
	const openaiRequest = {
		model: openaiModel,
		messages: messages,
		stream: !!anthropicRequest.stream,
	};

	// Map other parameters
	if (anthropicRequest.temperature !== undefined) {
		openaiRequest.temperature = anthropicRequest.temperature;
	}

	if (anthropicRequest.max_tokens !== undefined) {
		openaiRequest.max_tokens = anthropicRequest.max_tokens;
	}

	if (anthropicRequest.top_p !== undefined) {
		openaiRequest.top_p = anthropicRequest.top_p;
	}

	if (anthropicRequest.top_k !== undefined) {
		// OpenAI doesn't have top_k, so we ignore this parameter
	}

	return openaiRequest;
}

// Convert OpenAI response format to Anthropic format
function convertOpenAIToAnthropic(openaiResponse, originalRequest) {
	const content = openaiResponse.choices[0].message.content;

	// Build the Anthropic-style response
	const anthropicResponse = {
		id: `msg_${crypto.randomUUID().replace(/-/g, '')}`,
		type: 'message',
		role: 'assistant',
		content: content,
		model: originalRequest.model || 'claude-proxy', // Keep original model for compatibility
		stop_reason: mapStopReason(openaiResponse.choices[0].finish_reason),
		stop_sequence: null,
		usage: {
			input_tokens: openaiResponse.usage.prompt_tokens,
			output_tokens: openaiResponse.usage.completion_tokens,
		},
	};

	return anthropicResponse;
}

// Process streaming responses
async function processStream(openaiStream, writable) {
	const writer = writable.getWriter();
	const encoder = new TextEncoder();
	const decoder = new TextDecoder();

	let buffer = '';
	let messageId = `msg_${crypto.randomUUID().replace(/-/g, '')}`;
	let sentHeader = false;
	let contentAccumulator = '';

	try {
		const reader = openaiStream.getReader();

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			// Append new data to buffer and process
			buffer += decoder.decode(value, { stream: true });

			// Process complete SSE lines
			const lines = buffer.split('\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				if (!line.startsWith('data:')) continue;

				let data = line.slice(5).trim();
				if (data === '[DONE]') continue;

				try {
					const json = JSON.parse(data);
					const delta = json.choices[0]?.delta?.content || '';

					// Accumulate content for this chunk
					contentAccumulator += delta;

					// Send the header only once
					if (!sentHeader) {
						const header = {
							type: 'message_start',
							message: {
								id: messageId,
								type: 'message',
								role: 'assistant',
								content: '',
								model: 'claude-proxy',
							},
						};

						await writer.write(encoder.encode(`data: ${JSON.stringify(header)}\n\n`));
						sentHeader = true;
					}

					if (delta) {
						// Send content delta
						const contentDelta = {
							type: 'content_block_delta',
							delta: {
								type: 'text_delta',
								text: delta,
							},
							index: 0,
						};

						await writer.write(encoder.encode(`data: ${JSON.stringify(contentDelta)}\n\n`));
					}

					// Handle message completion
					if (json.choices[0]?.finish_reason) {
						const stopReason = mapStopReason(json.choices[0].finish_reason);

						const messageDone = {
							type: 'message_delta',
							delta: {
								stop_reason: stopReason,
							},
						};

						await writer.write(encoder.encode(`data: ${JSON.stringify(messageDone)}\n\n`));

						// Send final done message
						const done = {
							type: 'message_stop',
						};

						await writer.write(encoder.encode(`data: ${JSON.stringify(done)}\n\n`));
					}
				} catch (e) {
					console.error('Error parsing SSE data:', e);
				}
			}
		}
	} catch (e) {
		console.error('Stream processing error:', e);
	} finally {
		writer.close();
	}
}

// Map OpenAI finish reasons to Anthropic stop reasons
function mapStopReason(finishReason) {
	switch (finishReason) {
		case 'stop':
			return 'end_turn';
		case 'length':
			return 'max_tokens';
		case 'content_filter':
			return 'stop_sequence';
		default:
			return finishReason;
	}
}
