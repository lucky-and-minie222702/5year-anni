function loadTokenizerSync() {
	const request = new XMLHttpRequest();
	request.open("GET", "./tokenizer.json", false);
	request.send(null);

	if (request.status === 200) {
		const tokenizer = JSON.parse(request.responseText);
		return tokenizer;
	} else {
		throw new Error("Failed to load tokenizer");
	}
}

const tokenizer = loadTokenizerSync();

function text2ids(text) {
	const t = text.split(" ");
	let r = [];
	for (let i = 0; i < t.length; i++) {
		if (t[i] == "") continue;
		r.push(tokenizer.word_index[t[i]]);
	}
	return r;
}

function ids2text(id) {
	let r = [];
	for (let i = 0; i < id.length; i++) {
		if (id == 1) break;
		r.push(tokenizer.index_word[id[i]]);
	}
	return r.join(" ");
}

function sampleMultinomial(probabilities, n = 1) {
	const len = probabilities.length;
	const cumulativeProbs = new Float64Array(len);
	const results = new Uint32Array(n);

	let sum = 0;
	for (let i = 0; i < len; i++) {
		sum += probabilities[i];
		cumulativeProbs[i] = sum;
	}

	for (let j = 0; j < n; j++) {
		const r = Math.random();
		let low = 0;
		let high = len - 1;

		while (low < high) {
			const mid = (low + high) >>> 1;
			if (r > cumulativeProbs[mid]) {
				low = mid + 1;
			} else {
				high = mid;
			}
		}
		results[j] = low;
	}

	return results;
}

function softmax(logits, temperature = 0.5) {
	const adjustedLogits = logits.map((l) => l / temperature);

	const maxLogit = Math.max(...adjustedLogits);
	const scores = adjustedLogits.map((l) => Math.exp(l - maxLogit));
	const sum = scores.reduce((a, b) => a + b);

	return scores.map((s) => s / sum);
}

function cleanText(text) {
	text = text.toLowerCase().trim();

	text = text.replace(/:/g, " :");

	text = text.replace(/(?=[a-zA-Z])\d+|\d+(?=[a-zA-Z])/g, "");

	text = text.replace(/([a-zA-Z])\1+(?=\b)/g, "$1");

	text = text.replace(/❤+/g, "❤");

	text = text.replace(/áa/g, "á");
	text = text.replace(/ạa/g, "ạ");
	text = text.replace(/àa/g, "à");

	text = text.replace(/([^\s])([\u{1F300}-\u{1FAFF}❤])/gu, "$1 $2");

	text = text.replace(/cal/g, "call");

	text = text.replace(/❤️‍ 🔥/g, "❤️‍🔥");

	text = text.replace(/(?<!\u200D)❤️(?!\u200D)/gu, "❤️‍🔥");

	return text;
}

function toWord(index, tokenizer, skip_oov = False) {
	if (skip_oov && index == 1) {
		return "";
	}
	let w = tokenizer.index_word[index];
	if (w == undefined) return "";
	if (w.startsWith("3")) w = ":" + w;
	return w;
}

function hasRepeatingBigrams(ids) {
	const seen = new Set();
	let c = 0;

	for (let i = 0; i < ids.length - 1; i++) {
		const bigram = `${ids[i]}|${ids[i + 1]}`;

		if (seen.has(bigram)) {
			c++;
			console.log(bigram);
		}

		if (c > 2) return true;

		seen.add(bigram);
	}

	return false;
}

async function greedySearch(
	session,
	tokenizer,
	seedText,
	maxSteps,
	temperature
) {
	const stopIdx = 3;
	let currentSeq = text2ids(cleanText(seedText));
	const seedSeqLen = currentSeq.length;
	const maxLen = Math.min(64, seedSeqLen);

	for (let step = 0; step < maxSteps; step++) {
		const inputBuffer = new Int32Array(maxLen);
		const truncated = currentSeq.slice(-maxLen);
		const offset = maxLen - truncated.length;

		for (let i = 0; i < maxLen; i++) {
			inputBuffer[i] = i < truncated.length ? truncated[i] : 0;
		}

		const inputTensor = new ort.Tensor("int32", inputBuffer, [1, maxLen]);
		const results = await session.run({ input_ids: inputTensor });
		const outputData = results[Object.keys(results)[0]].data;

		let nextToken = 0;
		let maxLogit = -Infinity;

		const probs = softmax(outputData, temperature);
		nextToken = sampleMultinomial(probs, 1)[0];

		currentSeq.push(nextToken);

		if (hasRepeatingBigrams(currentSeq.slice(seedSeqLen))) {
			nextToken = stopIdx;
			currentSeq.push(nextToken);
		}

		if (nextToken === stopIdx) break;
	}

	if (currentSeq.at(-1) != stopIdx) currentSeq.push(stopIdx);

	return currentSeq
		.slice(seedSeqLen)
		.map((idx) => toWord(idx, tokenizer, true))
		.join(" ")
		.trim();
}

async function runInference(
	seedText,
	maxSteps,
	temperature,
	doSampleMultinomial = false
) {
	try {
		const session = await ort.InferenceSession.create("./model.onnx", {
			executionProviders: ["wasm"],
		});

		const results = greedySearch(
			session,
			tokenizer,
			seedText,
			maxSteps,
			temperature,
			doSampleMultinomial
		);
		console.log("Successfully run the model!");
		return results;
	} catch (e) {
		console.error("Failed to run model", e);
	}
}

function setChat(conversation) {
	const messages = conversation
		.replace(/<sender/g, "<sep> <sender")
		.trim()
		.split("<sep>")
		.map((msg) => msg.trim())
		.map((msg) => {
			const i = msg.indexOf(">");
			return {
				sender: msg.slice(0, i + 1).slice(1, -1),
				content: msg.slice(i + 2).trim(),
			};
		})
		.filter((msg) => msg.content.length > 0 && msg.sender.length > 0);

	let htmlOutput = messages
		.map(
			(msg) =>
				`
			<div class="message-wrapper">
				<div class="message ${msg.sender}">
					<div>${msg.content}</div>
				</div>
			</div>
		`
		)
		.join("");
	htmlOutput = `<div style = "height: 100%;"></div>${htmlOutput}`;

	document.getElementById("chat").innerHTML = htmlOutput;
	const element = document.getElementById("chat");
	element.scrollTop = element.scrollHeight;
}

// EVENT
const myForm = document.getElementById("main-form");

const resultOutput = document.getElementById("result");

const loader = document.getElementById("loading");
loader.classList.add("d-none");

let conversation = "<sendercattien>";

myForm.addEventListener("submit", function (event) {
	event.preventDefault();
	// seed

	const rawSeedValue = document.getElementById("seedText").value.trim();

	if (rawSeedValue.includes(":3") || rawSeedValue.includes("❤️‍🔥"))
		startHeartEffect(3);

	document.getElementById("seedText").value = "";
	const seedValue = conversation + " " + rawSeedValue + " <senderminhtriet>";
	conversation = seedValue;
	setChat(conversation);

	const maxSteps = 64;
	const temperature = parseFloat(temperatureRangeInput.value);
	// const temperature = 0.1;
	const doSampleMultinomial = true;

	if (rawSeedValue == "") {
		return;
	}

	loader.classList.remove("d-none");

	let promise = runInference(
		seedValue,
		maxSteps,
		temperature,
		doSampleMultinomial
	);
	promise.then((result) => {
		loader.classList.add("d-none");

		if (result.includes(":3") || result.includes("❤️‍🔥")) startHeartEffect(3);

		conversation = conversation.trim() + " " + result.trim();

		// show results
		setChat(conversation);
	});
});

// temperature
const temperatureRangeInput = document.getElementById("temperature-range");
const temperatureRangeOutput = document.getElementById(
	"temperature-range-value"
);

//
function startHeartEffect(durationInSeconds) {
	const container = document.getElementById("heart-container");

	// 1. Start the interval and store the ID so we can stop it later
	const heartInterval = setInterval(() => {
		const heart = document.createElement("div");
		heart.classList.add("heart");
		heart.innerHTML = "❤️‍🔥";

		heart.style.left = Math.random() * 100 + "vw";
		heart.style.animationDuration = Math.random() * 2 + 3 + "s";

		container.appendChild(heart);

		// Individual heart cleanup (removes from DOM after animation)
		setTimeout(() => {
			heart.remove();
		}, 3000);
	}, 300);

	// 2. The Timeout: Stop creating hearts after X seconds
	setTimeout(() => {
		clearInterval(heartInterval);
	}, durationInSeconds * 1000);
}

startHeartEffect(3);
