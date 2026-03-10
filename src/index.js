const { loadRF, getWasm } = require('./wasm.js')
const { RFModel: RFModelImpl } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const RFModel = createModelClass(RFModelImpl, RFModelImpl, { name: 'RFModel', load: loadRF })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await RFModel.create(params)
  await model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await RFModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadRF, getWasm, RFModel, train, predict }
