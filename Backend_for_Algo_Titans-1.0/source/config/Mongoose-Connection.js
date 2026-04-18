const mongoose = require('mongoose')
const config = require('config')
const dbgr = require('debug')("development:mongoose")

mongoose
    .connect(`${config.get("MONGODB_URI")}/Algo_Titans`)
    .then(() => {
        dbgr('successfully connected to database');
    })
    .catch((err) => {
        dbgr(err);
    })

module.exports = mongoose.connection  