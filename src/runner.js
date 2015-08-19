importScripts('recurrent.js');

// prediction params
var sample_softmax_temperature = 1.0; // how peaky model predictions should be
var max_chars_gen = 1000; // max length of generated sentences

// various global var inits
var epoch_size = -1;
var input_size = -1;
var output_size = -1;
var letterToIndex = {};
var indexToLetter = {};
var vocab = [];
var data_msgs = [];
var solver = new R.Solver(); // should be class because it needs memory for step caches

var model = {};

var initVocab = function(count_threshold) {
  // go over all characters and keep track of all unique ones seen
  // select 10% of conversations
  var txt = data_msgs.reduce(function(curr, nxt){return curr+nxt.join('');}, ''); // fixed input

  // count up all characters
  var d = {};
  for(var i=0,n=txt.length;i<n;i++) {
    var txti = txt[i];
    if(txti in d) { d[txti] += 1; } 
    else { d[txti] = 1; }
  }

  // filter by count threshold and create pointers
  var letterToIndex = {},
      indexToLetter = {},
      vocab = [];
  // NOTE: start at one because we will have START and END tokens!
  // that is, START token will be index 0 in model letter vectors
  // and END token will be index 0 in the next character softmax
  var q = 1; 
  for(ch in d) {
    if(d[ch] >= count_threshold) {
      // add character to vocab
      letterToIndex[ch] = q;
      indexToLetter[q] = ch;
      vocab.push(ch);
      q++;
    }
  }

  return {vocab: vocab, indexToLetter: indexToLetter, letterToIndex: letterToIndex, input_size: vocab.length + 1, output_size: vocab.length + 1, epoch_size: data_msgs.length};
}

var utilAddToModel = function(modelto, modelfrom) {
  for(var k in modelfrom) {
    if(modelfrom.hasOwnProperty(k)) {
      // copy over the pointer but change the key to use the append
      modelto[k] = modelfrom[k];
    }
  }
}

var saveModelInternal = function() {
  var out = {};
  out['hidden_sizes'] = hidden_sizes;
  out['generator'] = generator;
  out['letter_size'] = letter_size;
  var model_out = {};
  for(var k in model) {
    if(model.hasOwnProperty(k)) {
      model_out[k] = model[k].toJSON();
    }
  }
  out['model'] = model_out;
  out['letterToIndex'] = letterToIndex;
  out['indexToLetter'] = indexToLetter;
  out['vocab'] = vocab;
  return out;
}


var saveSovlerInternal = function(){
  var solver_out = {};
  solver_out['decay_rate'] = solver.decay_rate;
  solver_out['smooth_eps'] = solver.smooth_eps;
  step_cache_out = {};
  for(var k in solver.step_cache) {
    if(solver.step_cache.hasOwnProperty(k)) {
      step_cache_out[k] = solver.step_cache[k].toJSON();
    }
  }
  solver_out['step_cache'] = step_cache_out;
  return solver_out;
}

var loadModelInternal = function(j){
  hidden_sizes = j.hidden_sizes;
  generator = j.generator;
  letter_size = j.letter_size;
  model = {};
  for(var k in j.model) {
    if(j.model.hasOwnProperty(k)) {
      var matjson = j.model[k];
      model[k] = new R.Mat(1,1);
      model[k].fromJSON(matjson);
    }
  }
  letterToIndex = j['letterToIndex'];
  indexToLetter = j['indexToLetter'];
  vocab = j['vocab'];
}


var loadSolverInternal = function(j){
  solver = new R.Solver(); // have to reinit the solver since model changed
  solver.decay_rate = j.decay_rate;
  solver.smooth_eps = j.smooth_eps;
  solver.step_cache = {};
  for(var k in j.step_cache){
      if(j.step_cache.hasOwnProperty(k)){
          var matjson = j.step_cache[k];
          solver.step_cache[k] = new R.Mat(1,1);
          solver.step_cache[k].fromJSON(matjson);
      }
  }

}

var forwardIndex = function(G, model, ix, prev) {
  var x = G.rowPluck(model['Wil'], ix);
  // forward prop the sequence learner
  if(generator === 'rnn') {
    var out_struct = R.forwardRNN(G, model, hidden_sizes, x, prev);
  } else {
    var out_struct = R.forwardLSTM(G, model, hidden_sizes, x, prev);
  }
  return out_struct;
}

var predictConversation = function(context, max_char_gen, samplei, temperature) {
  if(typeof samplei === 'undefined') { samplei = false; }
  if(typeof temperature === 'undefined') { temperature = 1.0; }

  var G = new R.Graph(false);
  var output = '';
  var prev = {};

  for(i=0;i<context.length;i++){
    // Context
    var ix = i == 0 ? 0 : letterToIndex[context[i]];
    var lh = forwardIndex(G, model, ix, prev);
    prev = lh;
  }

  while(true) {
    // Generate

    // RNN tick
    var ix = output.length === 0 ? 0 : letterToIndex[output[output.length-1]];
    var lh = forwardIndex(G, model, ix, prev);
    prev = lh;

    // sample predicted letter
    logprobs = lh.o;
    if(temperature !== 1.0 && samplei) {
      // scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky
      for(var q=0,nq=logprobs.w.length;q<nq;q++) {
        logprobs.w[q] /= temperature;
      }
    }

    probs = R.softmax(logprobs);
    if(samplei) {
      var ix = R.samplei(probs.w);
    } else {
      var ix = R.maxi(probs.w);
    }
    
    if(ix === 0) break; // END token predicted, break out
    if(output.length > max_chars_gen) { break; } // something is wrong

    var letter = indexToLetter[ix];
    output += letter;
  }
  return {context: context, output: output};
}

var costfunconvo = function(model, msgs) {
  // takes a model and a message sequence and
  // calculates the loss. Also returns the Graph
  // object which can be used to do backprop
  var G = new R.Graph();
  var log2ppl = 0.0;
  var cost = 0.0;
  var prev = {};
  var totaln = 0;

  // Set context
  var currmsg = msgs[0],
      n = msgs[0].length;

  for(var i=-1;i<n;i++) {
    var ix_source = i === -1 ? 0 : letterToIndex[currmsg[i]]; // first: start with BREAK token
    lh = forwardIndex(G, model, ix_source, prev);
    prev = lh;

    // Note - no logging of error, as the NN can output whatever it likes at this point
  }

  var i;
  for(var m=1;m<msgs.length;m++) {
    currmsg = msgs[m];
    n = currmsg.length;
    totaln += n;

    // Iterate over the message sequence:
    for(i=-1;i<n;i++){
      // start and end tokens are zeros
      var ix_source = i === -1 ? 0 : letterToIndex[currmsg[i]]; // first step: start with BREAK token
      var ix_target = i === n-1 ? 0 : letterToIndex[currmsg[i+1]]; // last step: end with END token

      lh = forwardIndex(G, model, ix_source, prev);
      prev = lh;

      // set gradients into logprobabilities
      logprobs = lh.o; // interpret output as logprobs
      probs = R.softmax(logprobs); // compute the softmax probabilities

      log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
      cost += -Math.log(probs.w[ix_target]);

      // write gradients into log probabilities
      logprobs.dw = probs.w;
      logprobs.dw[ix_target] -= 1
    }
  }
  var ppl = Math.pow(2, log2ppl / (totaln - 1));
  return {'G':G, 'ppl':ppl, 'cost':cost, 'totaln': totaln};
}

var reduceModels = function(modelIttList){
  // reduceModels takes a list of {model, noItt}
  // where the models are identically shaped and have identical weights
  // (otherwise it doesn't make sense) and combine their differentials.
  // The noItt is used to normalize the differentials after reducing
  var outmodel = modelIttList[0].model;
  var sumItt = modelIttList.reduce(function(curr,nxt){return curr+nxt.noItt;}, 0);

  for(var k in outmodel) {
    if(outmodel.hasOwnProperty(k)) {
      for(var i=1;i<modelIttList.length;i++){
        var othermodel = modelIttList[i].model;
        for(var j=0;j<outmodel[k].dw.length;j++){
          outmodel[k].dw[j] += othermodel[k].dw[j];
        }
      }
    }
  }

  return {model: outmodel, noItt: sumItt};
}

onmessage = function(e){
  switch(e.data.type){
    case "ping": postMessage({type: "pong"});
                 break;
    // load a new dataset
    case "setDataSet": data_msgs = e.data.data; postMessage({type: "setDataSetDone"});
                        break;
    // calcVocab counts through the dataset and returns the vocab
    case "calcVocab": var rets = initVocab(1);
                      postMessage({type: "calcVocabDone", vocab: rets.vocab, indexToLetter: rets.indexToLetter, letterToIndex: rets.letterToIndex, input_size: rets.input_size, output_size: rets.output_size, epoch_size: rets.epoch_size});
                      break;
    // setModel gets a model object and loads the parameters into memory
    case "setModel": loadModelInternal(e.data.model);
                     postMessage({type: "setModelDone"});
                     break;
    // process gets a model and list of conversation indexes to process
    case "process": loadModelInternal(e.data.model);
                    var cost_struct = {};
                    var ppls = [], costs = [];
                    for(var i=0;i<e.data.ids.length;i++){
                      // stop long convo blowing up backprop
                      if(data_msgs[e.data.ids[i]].reduce(function(curr,nxt){return curr+nxt.length}, 0) > 2000) continue;
                      cost_struct = costfunconvo(model, data_msgs[e.data.ids[i]]);
                      // update model's dws:
                      cost_struct.G.backward();
                      ppls.push(cost_struct.ppl);
                      costs.push(cost_struct.cost);
                      cost_struct = {}; // GC
                    }
                    postMessage({type: "processDone", model: saveModelInternal(), ppls: ppls, costs: costs});
                    break;
    case "predict": loadModelInternal(e.data.model);
                    var rets = [];
                    for(var i=0;i<e.data.ids.length;i++){
                      rets.push(predictConversation(data_msgs[e.data.ids[i]][0], (e.data.maxLength ? 1000 : e.data.maxLength), true, e.data.temperature));
                    }
                    postMessage({type: "predictDone", predictions: rets, argmax: predictConversation(data_msgs[e.data.ids[0]][0], false)});
                    break;

  }
}
postMessage({type: "setup"});