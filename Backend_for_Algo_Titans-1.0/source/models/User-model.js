const mongoose = require('mongoose')
const UserSchema = mongoose.Schema({
    FullName: String,
    Email: String,
    Password: String,
    Chats:[{
        type:mongoose.Schema.Types.ObjectId,
        ref:"Chat-model"
    }],
})

module.exports = mongoose.model("UserModel", UserSchema)