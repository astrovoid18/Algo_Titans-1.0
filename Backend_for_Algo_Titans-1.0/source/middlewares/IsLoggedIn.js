const jwt = require('jsonwebtoken')

const isLoggedIn = (req, res, next) => {
    try {
    
        let token = req.cookies.token

        if (!token) {
            return res.send("You must be logged in!")
        }

        
        let decoded = jwt.verify(token, process.env.JWT_KEY)

        req.user = decoded

        next()
    } catch (err) {
        res.send("Invalid or expired token")
    }
}

module.exports = isLoggedIn