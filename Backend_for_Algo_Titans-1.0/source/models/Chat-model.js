const mongoose = require('mongoose')
const ChatSchema = mongoose.Schema({
    User: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User-Model",
        required: true
    },
    ChatData: {
        type: String,
        required: true,
        trim: true,
        maxlength: 500
    },
}, { timestamps: true }) 
module.exports = mongoose.model("ChatModel", ChatSchema)