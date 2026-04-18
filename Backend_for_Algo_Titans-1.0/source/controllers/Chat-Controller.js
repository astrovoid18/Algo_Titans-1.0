const ChatModel = require("../models/Chat-model")

module.exports.CreateChat = async (req, res) => {
    try {
        const { ChatData } = req.body

        const userId = req.user.id

        const chat = await ChatModel.create({
            User: userId,
            ChatData
        })

        res.send(chat)

    } catch (err) {
        res.status(500).send(err.message)
    }
}