const jwt = require('jsonwebtoken')

const GenerateToken = (user) => {
    console.log("JWT:", process.env.JWT_KEY)  

    return jwt.sign(
        { Email: user.Email, id: user._id },
        process.env.JWT_KEY,
        { expiresIn: "1d" }
    )
}

module.exports.GenerateToken = GenerateToken